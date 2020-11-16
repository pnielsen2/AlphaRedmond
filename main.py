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
import data_generator
import agents
import networks

import multiprocessing as mp

#test

#Transition = namedtuple('Transition',
#                        ('input', 'value', 'policy'))





def main():

    # display parameters
    display = False
    windowwidth = 1030
    windowheight = 830
    boardsize = 810

    # game parameters
    dimension = 9
    number_of_games = 1000





    squaresize = boardsize / dimension
    margin = (min([windowwidth, windowheight]) - boardsize) / 2

    # nontrivial initializations (will change in distant future
    displayer = None if display == False else game_displayer.GameWindow(windowwidth, windowheight, boardsize, dimension)

    network = networks.FastNetwork()
    rollouts = 100

    game_data = []
    value_data = []
    policy_data = []
    dtype = torch.FloatTensor

    # dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

    replaybuffer = data_generator.ReplayBuffer(5)


    processes = []

    for i in range(4):
        print("hi")
        generator = data_generator.DataGenerator(network, rollouts, replaybuffer)
        process = mp.Process(target = generator.generate_game)
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


    #pool = mp.Pool(mp.cpu_count())
    #queue = mp.Queue(4)
    #datagenerators = [data_generator.DataGenerator(network, rollouts, replaybuffer) for i in range(4)]

    #for datagenerator in datagenerators:
        #queue.put(datagenerator.generate_game())
    #pool.map(datagenerator.generate_game(),  range(6))
'''
    for i in range(number_of_games):

        gamesim = game_simulator.GameSim(1, dimension, displayer)
        game_agents = {"black": agents.NNAgent(network, rollouts, gamesim), "white": agents.NNAgent(network, rollouts, gamesim)}
        # {"black": agents.HumanAgent(), "white": agents.HumanAgent()}
        just_was_clicking = False
        run = True

        gamesim.run(game_agents)
        game_data.append(gamesim.input_history)
        value_data.append(gamesim.winner)
        policy_data.append(torch.nn.functional.normalize(gamesim.visit_count_list, p = 1))
        #print(game_data)
        #print(value_data)
        #print(policy_data)
        print(i)

        #input = torch.cat((
        #print(list(zip(game_data, value_data, policy_data)))
'''
if __name__ == '__main__': main()
