import pygame
from pygame.locals import *
import numpy as np
from multiprocessing import Process
import random

import torch

import time
import math
import copy

import game_displayer
import game_simulator
import agents
import networks

#test

Transition = namedtuple('Transition',
                        ('input', 'value', 'policy'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, game):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = game
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = []
        chosen_games = random.sample(self.memory, batch_size)
        for game in chosen_games:
            move_number = random.randrange(len(game))
            imput = game[0][max((move_number - 15,0)) : move_number + 1]
            if len(input) < 16:
                input = torch.cat((torch.zeros((16 - len(input), 2, 9, 9)), input))
            value = game[1]
            policy = game[2][move_number]
            batch.append((input, value, policy))
        batch = tuple(zip(*batch))
        #tuple of 32 inputs, tuple of 32 values, tuple of 32 policies
        inputs = torch.cat(batch[0]).view(batch_size,

        return batch

    def __len__(self):
        return len(self.memory)




def main():


    # display parameters
    display = True
    windowwidth = 1030
    windowheight = 830
    boardsize = 810

    # game parameters
    dimension = 9
    number_of_games = 100





    squaresize = boardsize / dimension
    margin = (min([windowwidth, windowheight]) - boardsize) / 2

    # nontrivial initializations (will change in distant future
    displayer = None if display == False else game_displayer.GameWindow(windowwidth, windowheight, boardsize, dimension)

    network = networks.FastNetwork()
    rollouts = 100
    [black_agent, white_agent] = [agents.NNAgent(network, rollouts), agents.NNAgent(network, rollouts)]

    game_data = []
    value_data = []
    policy_data = []
    dtype = torch.FloatTensor
    # dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

    for i in range(number_of_games):

        gamesim = game_simulator.GameSim(0, dimension)

        just_was_clicking = False
        run = True

        gamesim.run(displayer, [black_agent, white_agent])
        game_data.append(gamesim.input_history)
        value_data.append(gamesim.winner)
        policy_data.append(torch.nn.functional.normalize(gamesim.visit_count_list, p = 1))
        print(game_data)
        print(value_data)
        print(policy_data)
        print(i)

        input = torch.cat((
        print(list(zip(game_data, value_data, policy_data)))

if __name__ == '__main__': main()
