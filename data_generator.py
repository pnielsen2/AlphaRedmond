import torch
import game_simulator
import agents
import random
import game_displayer


class DataGenerator():
    def __init__(self, network, rollouts, replaybuffer, device):
        self.network = network
        self.rollouts = rollouts
        self.replaybuffer = replaybuffer
        self.device = device
        display = True
        self.displayer = None if display == False else game_displayer.GameWindow(1030, 830, 810, 9)

    def generate_games(self, number_of_games):
        for i in range(number_of_games):
            print("generating game")
            gamesim = game_simulator.GameSim(1, 9, self.displayer, self.device)
            agent = agents.NNAgent(self.network, self.rollouts, gamesim, self.device)
            visit_proportions = torch.zeros(1,82).to(self.device)

            while gamesim.winner == None:
                next_move = agent.get_action()
                visit_proportions = torch.cat((visit_proportions, agent.root.visit_counts.view(1,-1) / self.rollouts))
                #step and inform agent of the result
                if gamesim.step(next_move):
                    agent.update_root_node(next_move, gamesim)
                    self.displayer.redrawgamewindow(gamesim.current_player, gamesim.boardstate)

            input_history = gamesim.input_history
            policies = visit_proportions
            if gamesim.winner == "black":
                winner = 1
            elif gamesim.winner == "white":
                winner = -1

            self.replaybuffer.save_game((input_history, policies, winner))
            print(len(self.replaybuffer.games))
            print(gamesim.winner)

class ReplayBuffer():
    def __init__(self, capacity, device):
        self.device = device
        self.capacity = capacity
        self.games = []
        self.position = 0
        print("new instance")

    def save_game(self, game_tuple):
        if len(self.games) < self.capacity:
            self.games.append(None)
        self.games[self.position] = game_tuple
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        input = torch.tensor([]).to(self.device)
        policy_target = torch.tensor([]).to(self.device)
        value_target = torch.tensor([]).to(self.device)
        sampled_games = random.sample(range(len(self.games)), batch_size)
        for game in sampled_games:
            position_number = random.randrange(len(self.games[game][1]))
            input_slice = self.games[game][0][position_number:position_number + 8].view(16,9,9)
            player_indicator = torch.zeros(2,9,9).to(self.device)
            if position_number % 2 == 0:
                player_indicator[0,0] += 1
            else:
                player_indicator[0,1] += 1

            single_position_input = torch.cat((player_indicator, input_slice)).view(1,18,9,9)
            input = torch.cat((input, single_position_input))

            policy_slice = self.games[game][1][position_number]
            policy_target = torch.cat((policy_target, torch.unsqueeze(policy_slice,0)))

            value = self.games[game][2]
            if position_number % 2 == 0:
                value = self.games[game][2]
            else:
                value = -self.games[game][2]
            value_target = torch.cat((value_target,torch.tensor([value]).view(-1,1).to(self.device)))

        return input, policy_target, value_target
