import pygame
from pygame.locals import *
from pygame import gfxdraw
import numpy as np
from multiprocessing import Process
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import copy

#test

class GameWindow():
    def __init__(self, windowwidth, windowheight, boardsize, dimension):
        self.width = windowwidth
        self.height = windowheight
        self.boardsize = boardsize
        self.dimension = dimension
        self.squaresize = self.boardsize / self.dimension
        self.margin = (min([self.width, self.height]) - boardsize) / 2
        (self.mousex, self.mousey) = (-1,-1)

        self.running = True

        pygame.init()
        self.win = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Basic Pygame program')


    def get_intersection(self, location):
        return tuple([round(
        (coordinate - self.margin - self.squaresize / 2) / self.squaresize
        ) for coordinate in location])

    def get_location(self, intersection):
        return tuple([
        round(
        coordinate * self.squaresize + self. margin + self.squaresize / 2
        ) for coordinate in intersection])

    def snap(self, location):
        return self.get_location(self.get_intersection(location))

    def on_board(self,intersection):
        if -1<intersection[0]<self.dimension and -1<intersection[1]<self.dimension:
            return True
        else:
            return False

    def redrawgamewindow(self, toplay, black_intersections, white_intersections):

        # draw background
        self.win.fill((165,174,144))

        # draw board
        pygame.draw.rect(
        self.win,
        (235,179,85),
        (self.margin, self.margin, self.boardsize, self.boardsize)
        )

        # draws vertical lines
        for i in range(self.dimension):
            pygame.draw.line(
            self.win,
            (0,0,0),
            # starts at upper left corner and works its way right
            (
            self.margin + self.squaresize * (i + 0.5),
            self.margin + self.squaresize / 2
            ),
            # starts at bottom left corner and works its way right
            (
            self.margin + self.squaresize * (i + 0.5),
            self.margin + self.boardsize - self.squaresize / 2)
            )

            # draws horizontal lines
        for i in range(self.dimension):
            pygame.draw.line(
            self.win,
            (0,0,0),
            # starts in the top left corner and works its way down
            (
            self.margin + self.squaresize / 2,
            self.margin + self.squaresize * (i + 0.5)
            ),
            # starts in the top right corner and works its way down
            (
            self.margin + self.boardsize - self.squaresize / 2,
            self.margin + self.squaresize * (i + 0.5))
            )

        # sets color of ghost stone based on whose turn it is
        if toplay == 0:
            ghoststone_color = (0,0,0,127)
        elif toplay == 1:
            ghoststone_color = (255,255,255,127)

        self.snap([self.mousex, self.mousey])

        def drawstone(location, color):
            pygame.gfxdraw.filled_circle(
            self.win, location[0], location[1], round(self.squaresize / 2) - 1, color
            )
            # draws anti-aliased perimeter for ghost stone
            pygame.gfxdraw.aacircle(
            self.win, location[0], location[1], round(self.squaresize / 2) - 1, color
            )

        # draws ghost stone
        if self.on_board(self.get_intersection([self.mousex,self.mousey])):
            drawstone(self.snap([self.mousex,self.mousey]), ghoststone_color)

        # draws stones on board
        stonestart = time.time()
        for intersection in black_intersections:
            drawstone(self.get_location(intersection), (0,0,0))
        for intersection in white_intersections:
            drawstone(self.get_location(intersection), (255,255,255))
        stoneend = time.time()
        #print(stoneend-stonestart)
        pygame.display.update()

    def get_action(self):
        move_complete = False
        just_was_clicking = True
        while True:
            self.mousex, self.mousey = pygame.mouse.get_pos()

            if just_was_clicking == False:
                if pygame.mouse.get_pressed()[0] == True:
                    just_was_clicking = True
                    click_intersection = displayer.get_intersection((self.mousex, self.mousey))
                    #self.next_move = click_intersection
                    move_complete = True
            elif just_was_clicking == True:
                if pygame.mouse.get_pressed()[0] == False:
                    just_was_clicking = False

            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
            if self.running == False or move_complete == True:
                break
        return click_intersection

class Stone():
    def __init__(self, intersection, player):
        self.captured = False
        self.intersection = intersection
        self.player = player
        if self.player == 0:
            self.color = (0, 0, 0)
        if self.player == 1:
            self.color = (255, 255, 255)
    def draw(self, win):
        pygame.gfxdraw.filled_circle(win, self.x, self.y, self.radius, self.color)
        pygame.gfxdraw.aacircle(win, self.x, self.y, self.radius, self.color)

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
                    self.just_passed = False
                    self.board_history.append(ko_check_intersections)
                    [self.black_intersections, self.white_intersections] = candidate_intersections[:]
                    self.filled_intersections = self.black_intersections[:] + self.white_intersections[:]
                    self.boardstate = [double_clear[:], self.board_history[:]]

                    self.groups = [[[intersection for intersection in group] for group in color] for color in self.candidate_groups]
                    input_board = torch.zeros(1,2,9,9)
                    for intersection in self.black_intersections:
                        input_board[0,0][intersection] = 1
                    for intersection in self.white_intersections:
                        input_board[0,1][intersection] = 1
                    self.input_history = torch.cat((self.input_history, input_board))
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
        groups = copy.deepcopy(self.groups[color])
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

class HumanAgent():
    def __init__(self):
        self.color = []
        self.black_intersections = []
        self.white_intersections = []
        self.board_history = []

    def ponder(self, gamesim):
        self.color = gamesim.current_player
        self.black_intersections = gamesim.black_intersections
        self.white_intersections = gamesim.white_intersections


    def get_intersection(self, displayer, illegal_moves):
        move_complete = False
        just_was_clicking = True
        while True:
            displayer.mousex, displayer.mousey = pygame.mouse.get_pos()

            if just_was_clicking == False:
                if pygame.mouse.get_pressed()[0] == True:
                    just_was_clicking = True
                    click_intersection = displayer.get_intersection((displayer.mousex, displayer.mousey))
                    move_complete = True
            elif just_was_clicking == True:
                if pygame.mouse.get_pressed()[0] == False:
                    just_was_clicking = False
            displayer.redrawgamewindow(self.color, self.black_intersections, self.white_intersections)

            for event in pygame.event.get():
                if event.type == QUIT:
                    displayer.running = False
            if move_complete == True:
                return click_intersection
            if displayer.running == False:
                break

class RandomAgent():
    def __init__(self):
        pass

    def ponder(self, black_intersections, white_intersections, board_history, color):
        self.black_intersections = black_intersections
        self.white_intersections = white_intersections

    def get_intersection(self, displayer, illegal_moves):
        if random.random() < 1 / 81:
            return (9,0)
        return (random.randint(0,8), random.randint(0,8))

class SoftmaxAgent():
    def __init__(self):
        pass

    def ponder(self, black_intersections, white_intersections, board_history, color):
        move_signals = torch.randn(9,9)
        pass_signal = torch.randn(1)
        a = move_signals.view(-1)
        b = torch.cat((a, pass_signal))
        probs = torch.nn.functional.softmax(b,0)
        self.prob_dist = torch.distributions.Categorical(probs)
    def get_intersection(self, displayer, illegal_moves):
        sample = self.prob_dist.sample()
        x = (sample // 9).item()
        y = (sample % 9).item()
        return (x,y)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.blocks = 9
        self.conv_block_conv = nn.Conv2d(18, 64, 3, padding = 1)
        self.conv_block_batch_norm = nn.BatchNorm2d(64)

        self.resid_block_conv_1s = [nn.Conv2d(64, 64, 3, padding = 1) for i in range(self.blocks)]
        self.resid_block_batch_norm_1s = [nn.BatchNorm2d(64) for i in range(self.blocks)]

        self.resid_block_conv_2s = [nn.Conv2d(64, 64, 3, padding = 1) for i in range(self.blocks)]
        self.resid_block_batch_norm_2s = [nn.BatchNorm2d(64) for i in range(self.blocks)]

        self.policy_conv = nn.Conv2d(64, 2, 1)
        self.policy_batch_norm = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(162, 82)

        self.value_conv = nn.Conv2d(64, 1, 1)
        self.value_batch_norm = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(81, 64)
        self.value_fc2 = nn.Linear(64, 1)



    def resid_tower(self, input):
        for i in range(self.blocks):
            intermediate_activation = F.relu(self.resid_block_batch_norm_1s[i](self.resid_block_conv_1s[i](input)))
            block_output = F.relu(self.resid_block_batch_norm_2s[i](self.resid_block_conv_2s[i](intermediate_activation)) + input)
            input = block_output
        return block_output

    def forward(self, x):
        convolutional_block = F.relu(self.conv_block_batch_norm(self.conv_block_conv(x)))
        residual_tower = self.resid_tower(convolutional_block)
        policy = self.policy_fc(F.relu(self.policy_batch_norm(self.policy_conv(residual_tower))).view(-1))
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(F.relu(self.value_batch_norm(self.value_conv(residual_tower)).view(-1))))))

        return policy, value

class Node():
    def __init__(self, reset_data, probs):
        self.c_puct = 1.5
        self.reset_data = reset_data
        self.probs = probs
        self.visit_counts = torch.zeros(82)
        self.total_action_values = torch.zeros(82)
        self.mean_action_values = torch.zeros(82)
        self.u_values = self.c_puct * self.probs
        self.expanded_edges = {}

class NNAgent():
    def __init__(self, network):
        self.network = network
        self.c_puct = 1.5



    def get_network_output(self, gamesim):
        player_indicator = torch.zeros(1,2,9,9)
        player_indicator[0, gamesim.current_player] += 1
        input = gamesim.input_history[-8:]
        input = torch.cat((player_indicator, input)).view(1,18,9,9)
        policy, value = self.network(input)
        mask = torch.cat(((gamesim.input_history[-1][0] + gamesim.input_history[-1][1]).view(-1),torch.tensor([0.]))).type(torch.uint8)
        policy[mask] = float('-inf')
        return policy, value

    def expand_edge(self, current_node, chosen_move):
        next_gamesim = copy.deepcopy(current_node.gamesim)
        next_gamesim.step(chosen_move)
        current_node.expanded_edges[chosen_move] = Node(next_gamesim)


    def ponder(self, gamesim):
        print("pondering")
        policy, value = self.get_network_output(gamesim)
        probs = torch.nn.functional.softmax(policy, 0)
        mcts_gamesim = copy.deepcopy(gamesim)
        root = Node(mcts_gamesim.record(), probs)
        while torch.sum(root.visit_counts).item() < 100:
            current_node = root
            mcts_gamesim.set(root.reset_data)
            node_move_pairs = []
            print("root visit counts")
            print(root.visit_counts)

            while True:
                # choose the edge with the highest soft upper bound
                soft_upper_bound = current_node.mean_action_values + current_node.u_values
                chosen_move = torch.max(soft_upper_bound, 0)[1].item()
                if chosen_move not in current_node.expanded_edges:
                    # leaf
                    # expand node
                    #mcts_gamesim.set(copy.deepcopy(current_node.reset_data))
                    mcts_gamesim.set(current_node.reset_data)
                    #save = copy.deepcopy(current_node.reset_data)
                    if mcts_gamesim.step(self.intersection_from_move_number(chosen_move)):
                        #print(save[0])
                        #print(current_node.reset_data[0])
                        #print(save[0] == current_node.reset_data[0])
                        policy, value = self.get_network_output(mcts_gamesim)
                        if mcts_gamesim.game_over:
                            reward = mcts_gamesim.winner * (mcts_gamesim.current_player * 2 - 1)
                        else:
                            reward = value[0]

                        probs = torch.nn.functional.softmax(policy, 0)
                        current_node.expanded_edges[chosen_move] = Node(mcts_gamesim.record(), probs)
                        # update statistics on higher nodes
                        node_move_pairs.append((current_node, chosen_move))
                        for (node, move) in node_move_pairs:
                            node.visit_counts[move] += 1
                            node.total_action_values[move] += reward
                            node.mean_action_values[move] = node.total_action_values[move] / node.visit_counts[move]
                            node.u_values = self.c_puct * node.probs * torch.sqrt(1 + torch.sum(node.visit_counts)) / (1 + node.visit_counts)
                        break
                    else:
                        current_node.mean_action_values[chosen_move] = float(-1000)
                else:
                    # go to that node
                    node_move_pairs.append((current_node, chosen_move))
                    current_node = current_node.expanded_edges[chosen_move]



        self.visit_counts = root.visit_counts
        self.prob_dist = torch.distributions.Categorical(probs)

    def intersection_from_move_number(self, number):
        x = (number // 9)
        y = (number % 9)

        return (x,y)

    def get_intersection(self, displayer, illegal_moves, MCTS = True):
        if MCTS:
            visit_dist = torch.distributions.Categorical(self.visit_counts)
            move_output = visit_dist.sample().item()
        else:
            move_output = self.prob_dist.sample().item()
        #print(self.intersection_from_move_number(move_output))
        return self.intersection_from_move_number(move_output)



def main():

    # game parameters
    windowwidth = 1030
    windowheight = 830
    boardsize = 810
    dimension = 9
    display = True

    squaresize = boardsize / dimension
    margin = (min([windowwidth, windowheight]) - boardsize) / 2

    # nontrivial initializations (will change in distant future
    displayer = None if display == False else GameWindow(windowwidth, windowheight, boardsize, dimension)

    network = Network()
    [black_agent, white_agent] = [NNAgent(network), NNAgent(network)]

    #for i in range(10):
    gamesim = GameSim(0, dimension)

    just_was_clicking = False
    run = True
    gamesim.run(displayer, [black_agent, white_agent])

if __name__ == '__main__': main()
