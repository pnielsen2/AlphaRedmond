import pygame
from pygame.locals import *
import torch
import copy
import game_simulator


class HumanAgent():
    def __init__(self):
        self.color = []
        self.black_intersections = []
        self.white_intersections = []
        self.board_history = []

    def ponder(self, gamesim, displayer):
        self.color = gamesim.current_player
        self.black_intersections = gamesim.black_intersections
        self.white_intersections = gamesim.white_intersections


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
    def __init__(self, network, rollouts):
        self.network = network
        self.c_puct = 1.5
        self.mcts_gamesim = game_simulator.GameSim(0,9)
        self.dirichlet_noise = torch.distributions.dirichlet.Dirichlet(torch.zeros(82)+.13)
        self.rollouts = rollouts

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



    def get_network_output(self, gamesim):
        player_indicator = torch.zeros(1,2,9,9)
        player_indicator[0, gamesim.current_player] += 1
        input = gamesim.input_history[-8:]
        input = torch.cat((player_indicator, input)).view(1,18,9,9)
        policy, value = self.network(input)
        mask = torch.cat(((gamesim.input_history[-1][0] + gamesim.input_history[-1][1]).view(-1),torch.tensor([0.]))).type(torch.bool)
        policy[mask] = float('-inf')
        return policy, value


    def ponder(self, gamesim, displayer):
        #print("pondering")
        policy, value = self.get_network_output(gamesim)
        probs = torch.nn.functional.softmax(policy, 0)
        self.mcts_gamesim = copy.deepcopy(gamesim) # to remove later
        root = self.Node(self.mcts_gamesim.record(), (1 - .25) * probs + .25 * self.dirichlet_noise.sample())
        while torch.sum(root.visit_counts).item() < self.rollouts:
            current_node = root
            node_move_pairs = set([])
            while True:
                # choose the edge with the highest soft upper bound
                soft_upper_bound = current_node.mean_action_values + current_node.u_values
                chosen_move = torch.max(soft_upper_bound, 0)[1].item()
                if chosen_move not in current_node.expanded_edges:
                    # leaf
                    # expand node
                    #mcts_gamesim.set(copy.deepcopy(current_node.reset_data))
                    intersections = (current_node.reset_data[1] + current_node.reset_data[2])[:]
                    if len(intersections) != len(set(intersections)):
                        intersections.sort()
                        print("inside ponder (mcts, reset_data 1)")
                        print(intersections)
                    self.mcts_gamesim.set(current_node.reset_data)
                    #save = copy.deepcopy(current_node.reset_data)
                    intersections = (self.mcts_gamesim.black_intersections + self.mcts_gamesim.white_intersections)[:]
                    if len(intersections) != len(set(intersections)):
                        intersections.sort()
                        print("inside ponder (mcts, after reset)")
                        print(intersections)
                    intersections = (gamesim.black_intersections + gamesim.white_intersections)[:]
                    if len(intersections) != len(set(intersections)):
                        intersections.sort()
                        print("inside ponder (gamesim)")
                        print(intersections)
                    if self.mcts_gamesim.step(self.intersection_from_move_number(chosen_move)):
                        #print(save[0])
                        #print(current_node.reset_data[0])
                        #print(save[0] == current_node.reset_data[0])
                        intersections = (self.mcts_gamesim.black_intersections + self.mcts_gamesim.white_intersections)[:]
                        if len(intersections) != len(set(intersections)):
                            intersections.sort()
                            print("inside ponder (mcts, reset_data 4)")
                            print(intersections)
                        self.update_search_tree(current_node, chosen_move, node_move_pairs, self.mcts_gamesim.winner)
                        break
                    else:
                        intersections = (self.mcts_gamesim.black_intersections + self.mcts_gamesim.white_intersections)[:]
                        if len(intersections) != len(set(intersections)):
                            intersections.sort()
                            print("inside ponder (mcts, illegal move)")
                            print(intersections)
                        current_node.mean_action_values[chosen_move] = float(-100000)
                        current_node.total_action_values[chosen_move] = float(-100000)
                else:
                    node_move_pairs.add((current_node, chosen_move))
                    #check if terminal state
                    if current_node.expanded_edges[chosen_move].reset_data[7] != None:
                        reward = current_node.expanded_edges[chosen_move].reset_data[7]
                        node_move_pairs.add((current_node, chosen_move))
                        self.backup(node_move_pairs, reward)
                        break
                    else:
                        # go to that node
                        current_node = current_node.expanded_edges[chosen_move]
                #displayer.redrawgamewindow(current_node.reset_data[0], current_node.reset_data[1], current_node.reset_data[2])
            self.mcts_gamesim.set(root.reset_data)
        #print("root visit counts")
        #print(root.visit_counts)
        self.visit_counts = root.visit_counts
        self.prob_dist = torch.distributions.Categorical(probs)

    def update_search_tree(self, current_node, chosen_move, node_move_pairs, winner = None):
        # get reward and probs
        reward, probs = self.get_reward_and_probs(self.mcts_gamesim)
        # add node to search tree
        current_node.expanded_edges[chosen_move] = self.Node(self.mcts_gamesim.record(), probs)
        current_node.expanded_edges[chosen_move].winner = self.mcts_gamesim.winner
        # add current node and chosen move to rollout history
        node_move_pairs.add((current_node, chosen_move))
        # update statistics on higher nodes
        self.backup(node_move_pairs, reward)

    def get_reward_and_probs(self, gamesim):
        policy, value = self.get_network_output(gamesim)
        if gamesim.winner != None:
            reward = gamesim.winner * (gamesim.current_player * 2 - 1)
        else:
            reward = value[0]
        probs = torch.nn.functional.softmax(policy, 0)
        return (reward, probs)

    def backup(self, history, reward):
        for (node, move) in history:
            node.visit_counts[move] += 1
            node.total_action_values[move] += reward * self.mcts_gamesim.current_player * node.reset_data[0]
            node.mean_action_values[move] = node.total_action_values[move] / node.visit_counts[move]
            node.u_values = self.c_puct * node.probs * torch.sqrt(1 + torch.sum(node.visit_counts)) / (1 + node.visit_counts)

    def intersection_from_move_number(self, number):
        x = (number // 9)
        y = (number % 9)
        return (x,y)

    def get_intersection(self, gamesim, displayer, MCTS = True):
        if MCTS:
            if gamesim.moves_played < 8:
                visit_dist = torch.distributions.Categorical(self.visit_counts)
                move_output = visit_dist.sample().item()
            else:
                max_visit_count = self.visit_counts.max().item()
                max_indices = (self.visit_counts == max_visit_count).nonzero().view(-1)
                move_output = max_indices[torch.randint(len(max_indices),(1,))].item()
        else:
            move_output = self.prob_dist.sample().item()
        #print(self.intersection_from_move_number(move_output))
        return self.intersection_from_move_number(move_output)
