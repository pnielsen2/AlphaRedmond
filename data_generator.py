import torch
import game_simulator
import agents


class DataGenerator():
    def __init__(self, network, rollouts, replaybuffer):
        self.network = network
        self.rollouts = rollouts
        self.replaybuffer = replaybuffer

    def generate_game(self):
        for i in range(2):
            print("generating game")
            gamesim = game_simulator.GameSim(1, 9, None)
            agent = agents.NNAgent(self.network, self.rollouts, gamesim)
            visit_proportions = torch.zeros(1,82)

            while gamesim.winner == None:
                next_move = agent.get_action()
                visit_proportions = torch.cat((visit_proportions, agent.root.visit_counts.view(1,-1) / self.rollouts))
                #step and inform agent of the result
                if gamesim.step(next_move):
                    agent.update_root_node(next_move, gamesim)

            input_history = gamesim.input_history
            policies = visit_proportions
            if gamesim.winner == "black":
                winner = 1
            elif gamesim.winner == "white":
                winner = -1

            self.replaybuffer.save_game((input_history, policies, winner))
            print(len(self.replaybuffer.games))

class ReplayBuffer():
    def __init__(self, capacity):
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
        sampled_games = random.sample(range(len(self.games)), batch_size)
        for game in sampled_games:
            self.games[game]
