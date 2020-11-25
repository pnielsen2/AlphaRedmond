import torch
import pygame
from pygame.locals import *
import copy
import collections
import time
import parameters


class GameSim():
    def __init__(self, first_player, dimension, displayer, device):
        self.device = device
        self.current_player = "black"
        self.moves_played = 0
        self.filled_intersections = set([])
        self.groups = {"black": set([]), "white": set([])}
        self.dimension = dimension
        self.board_history = set([])
        self.input_history = torch.zeros(8,2,parameters.dimension, parameters.dimension).to(self.device)
        Board = collections.namedtuple('Board', ["black", "white"])
        #self.boardstate = Board(set([]), set([]))
        self.boardstate = (set([]), set([]))
        self.just_passed = False
        self.next_move = []
        self.running = True
        self.visit_count_list = torch.tensor([]).to(self.device)
        self.winner = None
        self.captured_stones = []
        self.most_recent_move = None
        self.displayer = displayer

    class Group():
        def __init__(self, stones, liberties):
            self.stones = stones
            self.liberties = liberties

        def add_stone(self, next_move, next_move_liberties):
            self.stones.add(next_move)
            self.liberties.remove(next_move)
            self.liberties = self.liberties.union(next_move_liberties)

    def state_copy(self, state):
        current_player = state[0]
        groups = {"black": set([]), "white": set([])}
        for player in ["black","white"]:
            for group in state[1][player]:
                copy_group = self.Group({stone for stone in group.stones}, {liberty for liberty in group.liberties})
                groups[player].add(copy_group)
        board_history = {(frozenset({stone for stone in boardstate[0]}),frozenset({stone for stone in boardstate[1]})) for boardstate in state[2]}
        input_history = state[3].clone()
        just_passed = state[4]
        winner = state[5]
        moves_played = state[6]
        boardstate = ({stone for stone in state[7][0]},{stone for stone in state[7][1]})
        return (current_player, groups, board_history, input_history, just_passed, winner, moves_played, boardstate)



    def set(self, reset_data):
        reset_data = self.state_copy(reset_data)
        self.current_player = reset_data[0] # done
        self.groups = reset_data[1]
        self.board_history = reset_data[2]
        self.input_history = reset_data[3]
        self.just_passed = reset_data[4] # done
        self.winner = reset_data[5] # done
        self.moves_played = reset_data[6] # done
        self.boardstate = reset_data[7]
        #print("mcts gamesim set")

    def record(self):
        #return (copy.deepcopy(self.current_player), copy.deepcopy(self.groups), copy.deepcopy(self.board_history), self.input_history.clone(), copy.deepcopy(self.just_passed), copy.deepcopy(self.winner), copy.deepcopy(self.moves_played), copy.deepcopy(self.boardstate))
        return self.state_copy((self.current_player, self.groups, self.board_history, self.input_history, self.just_passed, self.winner, self.moves_played, self.boardstate))

    def switch_current_player(self):
        if self.current_player == "black":
            self.current_player = "white"
        elif self.current_player == "white":
            self.current_player = "black"

    def on_board(self,intersection):
        if -1<intersection[0]<self.dimension and -1<intersection[1]<self.dimension:
            return True
        else:
            return False

    def checkIfDuplicates_1(self, listOfElems):
        if len(listOfElems) == len(set(listOfElems)):
            return False
        else:
            return True

    def update_input_history(self, next_move, removed_stones):

        input_board = torch.flip(self.input_history[-1:],[1])
        if next_move[0] != parameters.dimension:
            input_board[0,1][next_move] = 1
            for stone in removed_stones:
                input_board[0,0][stone] = 0

        self.input_history = torch.cat((self.input_history, input_board))

    def connect_groups(self, adjacent_groups, next_move, next_move_liberties):
        stones = set([])
        liberties = set([])
        stones.add(next_move)
        liberties = liberties.union(next_move_liberties)
        if len(adjacent_groups) > 0:
            for group in adjacent_groups:
                stones = stones.union(group.stones)
                liberties = liberties.union(group.liberties)
            liberties.remove(next_move)
        return self.Group(stones, liberties)

    def opposite_player(self, player):
        if player == "black":
            return "white"
        else:
            return "black"

    def player_id(self, player):
        if player == "black":
            return 0
        else:
            return 1


    def step(self, next_move):
        filled_intersections = self.boardstate[0].union(self.boardstate[1])
        #print("filled_intersections")
        #print(filled_intersections)

        removed_stones = []
        if next_move not in filled_intersections and self.on_board(next_move):
            # place move on board
            next_move_liberties = set([])
            for point in [(1,0),(0,1),(-1,0),(0,-1)]:
                adjacent_space = (next_move[0] + point[0], next_move[1] + point[1])
                if adjacent_space not in filled_intersections and self.on_board(adjacent_space):
                    next_move_liberties.add(adjacent_space)
            #print("current player")
            #print(self.current_player)
            self.boardstate[self.player_id(self.current_player)].add(next_move)
            #print("boardstate")
            #print(self.boardstate)
            # connect move to adjacent groups
            adjacent_groups = []
            for group in self.groups[self.current_player]:
                if next_move in group.liberties:
                    adjacent_groups.append(group)
            new_group = self.connect_groups(adjacent_groups, next_move, next_move_liberties)
            for group in adjacent_groups:
                self.groups[self.current_player].remove(group)
            self.groups[self.current_player].add(new_group)


            capture = False
            adjacent_opponent_groups = set([])
            captured_opponent_groups = set([])
            extra_liberties_due_to_capture = set([])
            for group in self.groups[self.opposite_player(self.current_player)]:
                if next_move in group.liberties:
                    group.liberties.remove(next_move)
                    adjacent_opponent_groups.add(group)
                if len(group.liberties) == 0:
                    capture = True
                    captured_opponent_groups.add(group)
                    #self.groups[self.opposite_player(self.current_player)].remove(group)
                    for stone in group.stones:
                        removed_stones.append(stone)
                        self.boardstate[self.player_id(self.opposite_player(self.current_player))].remove(stone)
                        #self.filled_intersections.remove(stone)
                        for group in self.groups[self.current_player]:
                            if any([self.adjacent(stone,current_player_stone) for current_player_stone in group.stones]):
                                group.liberties.add(stone)
                                extra_liberties_due_to_capture.add((group, stone))

            for group in captured_opponent_groups:
                self.groups[self.opposite_player(self.current_player)].remove(group)

            suicide = False
            if capture == False:
                if any([len(group.liberties) == 0 for group in self.groups[self.current_player]]):
                    suicide = True
                    # suicide, revert board state.
                    for group in adjacent_groups:
                        self.groups[self.current_player].add(group)
                    self.groups[self.current_player].remove(new_group)
                    for group in adjacent_opponent_groups:
                        group.liberties.add(next_move)
                    self.boardstate[self.player_id(self.current_player)].remove(next_move)

            if suicide == False:
                if (frozenset(self.boardstate[0]), frozenset(self.boardstate[1])) in self.board_history:
                    ko = True
                    # ko, revert board state.
                    for group in adjacent_groups:
                        self.groups[current_player].add(group)
                    self.groups[current_player].remove(group)
                    for group in captured_opponent_groups:
                        self.groups[self.opposite_player(self.current_player)].add(group)
                    for group in adjacent_opponent_groups:
                        group.liberties.add(next_move)
                    for group, liberty in extra_liberties_due_to_capture:
                        group.liberties.remove(liberty)
                    self.boardstate[self.player_id(self.current_player)].remove(next_move)
                    for stone in removed_stones:
                        self.boardstate[self.player_id(self.opposite_player(self.current_player))].remove(stone)
                else:
                    ko = False

                if not (suicide or ko):
                    self.just_passed = False
                    self.board_history.add((frozenset(self.boardstate[0]), frozenset(self.boardstate[0])))

                    self.update_input_history(next_move, removed_stones)
                    self.switch_current_player()
                    return True
                else:
                    return False

        elif next_move == (self.dimension, 0):
            if self.just_passed:
                #print("2nd pass")
                self.score()
            else:
                #print("pass")
                self.just_passed = True
            self.update_input_history(next_move, removed_stones)
            self.switch_current_player()



            return True
        else:
            #print("hi")
            #print(next_move)
            #print(next_move in filled_intersections)
            #print(self.on_board(next_move))
            return False

    def score(self):
        black_score = len(self.boardstate[0])
        white_score = len(self.boardstate[1]) + 7.5
        territory = [(x,y) for x in range(self.dimension) for y in range(self.dimension)]
        intersections = self.boardstate[0].union(self.boardstate[1])
        #intersections.sort()
        #print(intersections)
        for intersection in intersections:
            territory.remove(intersection)
        territory = [[element] for element in territory]
        empty_regions = self.group(territory)
        for empty_region in empty_regions:
            reaches = self.liberties(empty_region, territory)
            if all([bounding_stone in self.boardstate[0] for bounding_stone in reaches]):
                black_score += len(empty_region)
            if all([bounding_stone in self.boardstate[1] for bounding_stone in reaches]):
                white_score += len(empty_region)
        self.black_score = black_score
        self.white_score = white_score
        if black_score > white_score:
            self.winner = "black"
        else:
            self.winner = "white"

    # Takes a list of candidate groups and combines candidates that are part of
    # the same group.
    def group(self, l):
        done = False
        while True:
            match = False
            # basically, for each pair of groups, if any of their stones are
            # adjacent, combine the groups.
            for groupindex1 in range(len(l) - 1):
                for groupindex2 in range(groupindex1 + 1, len(l)):
                    if any([any([self.adjacent(coord1, coord2) for coord2 in l[groupindex2]]) for coord1 in l[groupindex1]]):
                        match = True
                        l[groupindex1] += l[groupindex2]
                        del l[groupindex2]
                        break
                if match == True:
                    break
            if match == False:
                break
        return l


    def adjacent(self, intersection1, intersection2):
        if intersection2 in set([(intersection1[0] - 1, intersection1[1]),
         (intersection1[0] + 1, intersection1[1]),
         (intersection1[0], intersection1[1] - 1),
         (intersection1[0], intersection1[1] + 1)]):
            return True
        else:
            return False

    def liberties(self, group, combined_intersections):
        candidates = set([])
        liberties = set([])
        for stone in group:
            candidates.add((stone[0] + 1, stone[1]))
            candidates.add((stone[0] - 1, stone[1]))
            candidates.add((stone[0], stone[1] + 1))
            candidates.add((stone[0], stone[1] - 1))
        for candidate in candidates:
            if candidate not in combined_intersections and self.on_board(candidate):
                liberties.add(candidate)
        return liberties

    def clear(self, filled_intersections, color):
        # black and white is a pair containing a list of
        intersections = (filled_intersections[0] + filled_intersections[1])[:]
        if len(intersections) != len(set(intersections)):
            intersections.sort()
            print("inside clear 4")
            print(intersections)
        black_and_white = filled_intersections[:]
        intersections = (black_and_white[0] + black_and_white[1])[:]
        if len(intersections) != len(set(intersections)):
            intersections.sort()
            print("inside clear 3")
            print(intersections)
        toclear = black_and_white[color]
        combined_intersections = black_and_white[0][:] + black_and_white[1][:]
        # the idea is that you don't need to re-calculate what the groups are
        # for the opponent's stones. They shouldn't have changed from the last
        # board state if they haven't played a move since they were last calculated.
        groups = [[intersection for intersection in group] for group in self.groups[color]]
        #groups = copy.deepcopy(self.groups[color])
        if color != self.current_player:
            clearedcolor = []
            candidate_groups = []
            # checks each group for liberties. Appends the stones to the list of
            # cleared stones and appends the group to the list of cleared groups
            # in that case.
            for i in range(len(groups)):
                if len(self.liberties(groups[i], combined_intersections)) > 0:
                    clearedcolor += groups[i]
                    candidate_groups.append(groups[i][:])
                else:
                    self.captured_stones += groups[i]
            self.candidate_groups[color] = candidate_groups
            black_and_white[color] = clearedcolor
            intersections = (black_and_white[0] + black_and_white[1])[:]
            if len(intersections) != len(set(intersections)):
                intersections.sort()
                print("inside clear 2")
                print(intersections)
            return black_and_white[:]
        # If the current player is being cleared, their groups can be calculated
        # by running the grouping algorithm on their existing groups, plus the
        # next move
        elif color == self.current_player:
            groups = self.group(groups + [[self.candidate_move]])
            # if any of the current player's groups have no liberties, they must
            # have made a suicide so clear returns a bogus boardstate which will
            # not match the previous clear of the opponent's stones
            if any([len(self.liberties(groups[i], combined_intersections)) == 0 for i in range(len(groups))]):
                return [[(-1,-1)],[(-2,-2)]]
            else:
                self.candidate_groups[color] = groups[:]
                intersections = (black_and_white[0] + black_and_white[1])[:]
                if len(intersections) != len(set(intersections)):
                    intersections.sort()
                    print("inside clear 1")
                    print(intersections)
                return black_and_white[:]
        #else:
            #exception


    def run(self, agents):
        while self.winner == None:
            #print(self.black_intersections)
            #print(self.white_intersections)
            agent = agents[self.current_player]
            self.next_move = agent.get_action()
            self.visit_count_list = torch.cat((self.visit_count_list, agent.root.visit_counts.view(1,-1)))

            #step and inform all agents of the result
            if self.step(self.next_move):
                agents["black"].update_root_node(self.next_move, self)
                agents["white"].update_root_node(self.next_move, self)

            if self.next_move != None:
                if self.displayer != None:
                    self.displayer.redrawgamewindow(self.current_player, self.boardstate[0], self.boardstate[1])

                    for event in pygame.event.get():
                        if event.type == QUIT:
                            self.displayer.running = False
                    if self.displayer.running == False:
                        break
            else:
                break
        print(self.black_score)
        print(self.white_score)
        print(self.winner)

    def get_action(self, agent):
        if self.most_recent_move != None:
            agent.update_root_node(self.most_recent_move)
        agent.ponder()

        #self.visit_count_list = torch.cat((self.visit_count_list, agent.visit_counts.view(1,-1)))
        self.most_recent_move = agent.get_intersection()
        return self.most_recent_move
        '''
        while True:
            intersection = agent.get_intersection(self, displayer)
            if intersection != None:
                legal = self.step(intersection)
                if legal:
                    break
                else:
            else:
                break
        return intersection
        '''
