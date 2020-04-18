import torch
import pygame
from pygame.locals import *
import copy

class GameSim():
    def __init__(self, first_player, dimension):
        self.current_player = first_player
        self.black_intersections = []
        self.white_intersections = []
        self.filled_intersections = []
        self.groups = [[],[]]
        self.candidate_groups = [[],[]]
        self.dimension = dimension
        self.board_history = [(set([]),set([]))]
        self.input_history = torch.zeros(8,2,9,9)
        self.boardstate = [[[], []], [(set([]),set([]))]]
        self.just_passed = False
        self.next_move = []
        self.running = True
        self.game_over = False

    def set(self, reset_data):
        self.current_player = reset_data[0]
        self.black_intersections = reset_data[1]
        self.white_intersections = reset_data[2]
        self.groups = reset_data[3]
        self.board_history = reset_data[4]
        self.input_history = reset_data[5]
        self.just_passed = reset_data[6]
        self.game_over = reset_data[7]

    def record(self):
        return [self.current_player, self.black_intersections, self.white_intersections, self.groups, self.board_history, self.input_history, self.just_passed, self.game_over]

    def switch_current_player(self):
        self.current_player = 1 - self.current_player

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

    def update_gamesim(self, board_hash, intersection_lists):
        self.just_passed = False
        self.board_history.append(board_hash)
        [self.black_intersections, self.white_intersections] = intersection_lists[:]
        self.filled_intersections = self.black_intersections[:] + self.white_intersections[:]
        self.boardstate = [intersection_lists[:], self.board_history[:]]

        self.groups = self.candidate_groups
        self.groups = [[[intersection for intersection in group] for group in color] for color in self.candidate_groups]

    def update_input_history(self):
        input_board = torch.zeros(1,2,9,9)
        for intersection in self.black_intersections:
            input_board[0,0][intersection] = 1
        for intersection in self.white_intersections:
            input_board[0,1][intersection] = 1
        self.input_history = torch.cat((self.input_history, input_board))

    def step(self, next_move):
        self.candidate_move = next_move
        if next_move not in self.filled_intersections and self.on_board(next_move):
            filled_intersections = [self.black_intersections[:], self.white_intersections[:]]
            filled_intersections[self.current_player].append(next_move)
            candidate_intersections = self.clear(filled_intersections, 1 - self.current_player)
            # removes any of the current player's stones with no liberties
            double_clear = self.clear(candidate_intersections, self.current_player)
            # if any of the currnt player's stones had no liberties and were
            # removed after clearing the opponents' stones, it must have been a
            # suicide, so we don't continue
            if set(double_clear[self.current_player]) == set(candidate_intersections[self.current_player]):
                # creates the current board position
                ko_check_intersections = (set(candidate_intersections[0][:]), set(candidate_intersections[1][:]))
                # checks if the current board is in the board history
                if not ko_check_intersections in self.board_history:
                    # all clear. Updates board history and intersections to the new state
                    self.update_gamesim(ko_check_intersections, double_clear)
                    self.update_input_history()
                    # changes to the opposing player's turn
                    self.switch_current_player()
                    return True

        elif next_move == (self.dimension, 0):
            if self.just_passed:
                self.score()
                self.game_over = True
            else:
                self.just_passed = True
            self.switch_current_player()
            return True
        else:
            return False


    def score(self):
        black_score = len(self.black_intersections)
        white_score = len(self.white_intersections) + 7.5
        territory = [(x,y) for x in range(self.dimension) for y in range(self.dimension)]
        for intersection in self.black_intersections:
            territory.remove(intersection)
        for intersection in self.white_intersections:
            territory.remove(intersection)
        territory = [[element] for element in territory]
        empty_regions = self.group(territory)
        for empty_region in empty_regions:
            reaches = self.liberties(empty_region, territory)
            if all([bounding_stone in self.black_intersections for bounding_stone in reaches]):
                black_score += len(empty_region)
            if all([bounding_stone in self.white_intersections for bounding_stone in reaches]):
                white_score += len(empty_region)
        self.black_score = black_score
        self.white_score = white_score
        self.winner = (black_score > white_score)*2-1

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
        if intersection2 in [(intersection1[0] - 1, intersection1[1]),
         (intersection1[0] + 1, intersection1[1]),
         (intersection1[0], intersection1[1] - 1),
         (intersection1[0], intersection1[1] + 1)]:
            return True
        else:
            return False

    def liberties(self, group, combined_intersections):
        candidates = []
        liberties = []
        for stone in group:
            candidates.append((stone[0] + 1, stone[1]))
            candidates.append((stone[0] - 1, stone[1]))
            candidates.append((stone[0], stone[1] + 1))
            candidates.append((stone[0], stone[1] - 1))
        for candidate in candidates:
            if all([candidate not in combined_intersections,
            candidate not in liberties, self.on_board(candidate)]):
                liberties.append(candidate)
        return liberties

    def clear(self, filled_intersections, color):
        # black and white is a pair containing a list of
        black_and_white = filled_intersections[:]
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
            self.candidate_groups[color] = candidate_groups
            black_and_white[color] = clearedcolor
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
                return [[(-1,-1)],[(-1,-1)]]
            else:
                self.candidate_groups[color] = groups[:]
                return black_and_white[:]
        #else:
            #exception


    def run(self, displayer, agents):
        while self.game_over == False:
            self.next_move = self.get_action(displayer, agents[self.current_player])

            if self.next_move != None:
                if displayer != None:
                    displayer.redrawgamewindow(self.current_player, self.black_intersections, self.white_intersections)

                    for event in pygame.event.get():
                        if event.type == QUIT:
                            displayer.running = False
                    if displayer.running == False:
                        break
            else:
                break
        print(self.black_score)
        print(self.white_score)
        print(self.winner)

    def get_action(self, displayer, agent):
        illegal_moves = []
        agent.ponder(self)
        while True:
            intersection = agent.get_intersection(displayer, illegal_moves)
            if intersection != None:
                legal = self.step(intersection)
                if legal:
                    break
            else:
                break
        return intersection
