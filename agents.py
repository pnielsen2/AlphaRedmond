import pygame
from pygame.locals import *
import torch
import copy
import game_simulator
import time


class HumanAgent():
    def __init__(self):
        self.color = []
        self.black_intersections = []
        self.white_intersections = []
        self.board_history = []

    def ponder(self, gamesim, displayer):
        self.color = gamesim.current_player
        self.black_intersections = gamesim.boardstate[0]
        self.white_intersections = gamesim.boardstate[1]


    def get_intersection(self, gamesim, displayer, illegal_moves):
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

    def get_intersection(self, gamesim, displayer, illegal_moves):
        if random.random() < 1 / 81:
            return (9,0)
        return (random.randint(0,8), random.randint(0,8))

class SoftmaxAgent():
    def __init__(self):
        pass

    def ponder(self, gamesim, displayer):
        move_signals = torch.randn(9,9)
        pass_signal = torch.randn(1)
        a = move_signals.view(-1)
        b = torch.cat((a, pass_signal))
        probs = torch.nn.functional.softmax(b,0)
        self.prob_dist = torch.distributions.Categorical(probs)

    def get_intersection(self, gamesim, displayer, illegal_moves):
        sample = self.prob_dist.sample()
        x = (sample // 9).item()
        y = (sample % 9).item()
        return (x,y)


class NNAgent():
    def __init__(self, network, rollouts, gamesim):
        self.network = network
        self.c_puct = 1.5
        #self.mcts_gamesim = game_simulator.GameSim(0,9)
        self.dirichlet_noise = torch.distributions.dirichlet.Dirichlet(torch.zeros(82)+.13)
        self.rollouts = rollouts
        self.gamesim = gamesim
        policy, value = self.get_network_output(gamesim.current_player, gamesim.input_history)
        probs = torch.nn.functional.softmax(policy, 0)
        self.mcts_gamesim = game_simulator.GameSim(1, gamesim.dimension, gamesim.displayer) # to remove later
        self.root = self.Node(self.mcts_gamesim.record(), (1 - .25) * probs + .25 * self.dirichlet_noise.sample())
        #self.root = self.Node(self.mcts_gamesim.record(), (1 - .25) * probs + .25 * self.dirichlet_noise.sample())

        self.prob_dist = torch.distributions.Categorical(probs)

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

            self.moves_played = self.reset_data[6]

    def get_action(self):
        self.ponder()

        #self.visit_count_list = torch.cat((self.visit_count_list, agent.visit_counts.view(1,-1)))
        return self.get_intersection()


    def get_network_output(self, current_player, input_history):
        player_indicator = torch.zeros(1,2,9,9)
        if current_player == "black":
            player_indicator[0,0] += 1
        else:
            player_indicator[0,1] += 1
        input = input_history[-8:]
        input = torch.cat((player_indicator, input)).view(1,18,9,9)
        policy, value = self.network(input)
        mask = torch.cat(((input_history[-1][0] + input_history[-1][1]).view(-1),torch.tensor([0.]))).type(torch.bool)
        #print(torch.sum(gamesim.input_history[-1][0]) + torch.sum(gamesim.input_history[-1][1]))
        #print("hi 1")
        policy[mask] = float('-inf')
        return policy, value

    def update_root_node(self, move, gamesim):
        if move in self.root.expanded_edges:
            self.root = self.root.expanded_edges[move]
        else:
            reward, probs = self.get_value_and_probs(gamesim.current_player, gamesim.input_history, gamesim.winner)
            self.root.expanded_edges[move] = self.Node(gamesim.record(), probs)
            self.root.expanded_edges[move].winner = gamesim.winner
            self.root = self.root.expanded_edges[move]

    def ponder(self):
        #print(self.gamesim.current_player)
        #print(torch.sum(self.root.visit_counts).item())
        #print("pondering")

        while torch.sum(self.root.visit_counts).item() < self.rollouts:
            depth = 0
            current_node = self.root
            #print("root board state")
            #print(self.root.reset_data[7])
            node_move_pairs = set([])
            #time.sleep(1)
            #print("new rollout")
            while True:
                # choose the edge with the highest soft upper bound
                soft_upper_bound = current_node.mean_action_values + current_node.u_values
                #print("soft upper bound")
                #print(soft_upper_bound)
                chosen_move = torch.max(soft_upper_bound, 0)[1].item()
                if chosen_move not in current_node.expanded_edges:
                    # leaf
                    # expand node
                    #mcts_gamesim.set(copy.deepcopy(current_node.reset_data))
                    #print("boardstate 1")
                    #print(current_node.reset_data[7])
                    #print("root board state 1")
                    #print(self.root.reset_data[7])
                    self.mcts_gamesim.set(current_node.reset_data)    # should have deepcopy!
                    #print("root board state 2")
                    #print(self.root.reset_data[7])
                    #save = copy.deepcopy(current_node.reset_data)
                    #print("chosen move")
                    #print(chosen_move)
                    if self.mcts_gamesim.step(self.intersection_from_move_number(chosen_move)):
                        #self.gamesim.displayer.redrawgamewindow(self.gamesim.current_player, self.gamesim.boardstate[0], self.gamesim.boardstate[1])
                        #self.gamesim.displayer.redrawgamewindow(self.mcts_gamesim.current_player, self.mcts_gamesim.boardstate[0], self.mcts_gamesim.boardstate[1])
                        #print(save[0])
                        #print(current_node.reset_data[0])
                        #print(save[0] == current_node.reset_data[0])
                        #print("root board state 3")
                        #print(self.root.reset_data[7])
                        #print("rollouts")
                        #print(torch.sum(self.root.visit_counts).item())
                        #print("depth: " + str(depth))
                        self.update_search_tree(current_node, chosen_move, node_move_pairs, self.mcts_gamesim.current_player, self.mcts_gamesim.input_history, self.mcts_gamesim.winner)
                        #print("root board state 4")
                        #print(self.root.reset_data[7])
                        break
                    else:
                        #print("step failed")
                        #print("location 1")
                        #print("illegal move")
                        current_node.mean_action_values[chosen_move] = float('-inf')
                        current_node.total_action_values[chosen_move] = float('-inf')
                else:
                    node_move_pairs.add((current_node, chosen_move))
                    #check if terminal state
                    if current_node.expanded_edges[chosen_move].reset_data[5] != None:
                        reward = (1 - 2 * self.gamesim.player_id(current_node.expanded_edges[chosen_move].reset_data[5])) * (1 - 2 * self.gamesim.player_id(current_node.reset_data[0]))
                        node_move_pairs.add((current_node, chosen_move))
                        self.backup(node_move_pairs, reward, self.mcts_gamesim.current_player)
                        break
                    else:
                        # go to that node
                        depth +=1
                        current_node = current_node.expanded_edges[chosen_move]
                #displayer.redrawgamewindow(current_node.reset_data[0], current_node.reset_data[1], current_node.reset_data[2])
            self.mcts_gamesim.set(self.root.reset_data)
        #print("root visit counts")
        #print(self.root.visit_counts)
        #print("root mean action values")
        #print(self.root.mean_action_values)
        self.visit_counts = self.root.visit_counts


    def update_search_tree(self, current_node, chosen_move, node_move_pairs, current_player, input_history, winner = None):
        # get reward and probs
        value, probs = self.get_value_and_probs(current_player, input_history, winner)
        # add node to search tree
        current_node.expanded_edges[chosen_move] = self.Node(self.mcts_gamesim.record(), probs)
        current_node.expanded_edges[chosen_move].winner = self.mcts_gamesim.winner
        # add current node and chosen move to rollout history
        node_move_pairs.add((current_node, chosen_move))
        # update statistics on higher nodes
        self.backup(node_move_pairs, value, current_player)

    def get_value_and_probs(self, current_player, input_history, winner):
        policy, value = self.get_network_output(current_player, input_history)
        if winner != None:
            value = 1 if winner == current_player else -1
        else:
            # value is from the perspective of the current player
            value = value.item()

        probs = torch.nn.functional.softmax(policy, 0)
        return (value, probs)

    def backup(self, history, value, current_player):
        #print("new backup")
        #print("backing up")
        for (node, move) in history:
            #print(move)
            node.visit_counts[move] += 1
            #print(node.reset_data[0])
            #print("reward")
            #print(reward * (1 - 2 * self.gamesim.player_id(node.reset_data[0])) * (1 - 2 * self.gamesim.player_id(self.gamesim.opposite_player(current_player))))
            # node reset data[0] is the current player in that node. This is
            # positive if the node's current player is the same as the player
            # who recieved the value.

            node.total_action_values[move] += value * (1 - 2 * self.gamesim.player_id(node.reset_data[0])) * (1 - 2 * self.gamesim.player_id(current_player))
            node.mean_action_values[move] = node.total_action_values[move] / node.visit_counts[move]
            node.u_values = self.c_puct * node.probs * torch.sqrt(1 + torch.sum(node.visit_counts)) / (1 + node.visit_counts)

    def intersection_from_move_number(self, number):
        x = (number // 9)
        y = (number % 9)
        return (x,y)

    def get_intersection(self, MCTS = True):
        if MCTS:
            if self.root.moves_played < 8:
                visit_dist = torch.distributions.Categorical(self.root.visit_counts)
                move_output = visit_dist.sample().item()
            else:
                max_visit_count = self.root.visit_counts.max().item()
                max_indices = (self.root.visit_counts == max_visit_count).nonzero().view(-1)
                move_output = max_indices[torch.randint(len(max_indices),(1,))].item()
        else:
            move_output = self.prob_dist.sample().item()
        #print(move_output)
        #print(self.intersection_from_move_number(move_output))
        return self.intersection_from_move_number(move_output)
