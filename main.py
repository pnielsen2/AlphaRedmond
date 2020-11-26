import pygame
from pygame.locals import *
import numpy as np
from multiprocessing import Process
import random

import torch
import torch.nn as nn
import torch.optim as optim

import time
import math
import copy

import game_displayer
import game_simulator
import data_generator
import agents
import networks

import multiprocessing as mp

import parameters


print("Number of processors: ", mp.cpu_count())
# nontrivial initializations (will change in distant future
displayer = None if parameters.display == False else game_displayer.GameWindow(parameters.windowwidth, parameters.windowheight, parameters.boardsize, parameters.dimension)

device = torch.device("cuda:0" if torch.cuda.is_available() and parameters.device =='gpu' else "cpu")

def main():

    network = networks.Network().to(device)
    print(next(network.parameters()).is_cuda)

    game_data = []
    value_data = []
    policy_data = []

    print(device)

    replaybuffer = data_generator.ReplayBuffer(parameters.replay_buffer_size, device)

    generator = data_generator.DataGenerator(network, parameters.rollouts, replaybuffer, device, displayer)



    policy_loss = nn.KLDivLoss(reduction = 'batchmean')
    value_loss = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr = parameters.learning_rate)

    policy_activation = nn.LogSoftmax()


    while True:
        generator.generate_games(parameters.games_generated_at_a_time)
        if len(replaybuffer.games) >= parameters.batch_size:
            for epoch in range(batches_per_train_loop):
                input, policy_target, value_target = replaybuffer.sample(parameters.batch_size)

                optimizer.zero_grad()

                policy_guess, value_guess = network(input)
                '''
                print("policy_guess")
                print(policy_guess)
                print("policy_target")
                print(policy_target)
                print("value_guess")
                print(value_guess)
                print("value_target")
                print(value_target)
                print((policy_guess, value_guess))
                '''
                loss1 = policy_loss(policy_activation(policy_guess), policy_target)
                loss2 = value_loss(value_guess, value_target)
                loss = loss1 + loss2
                '''
                print("policy_loss")
                print(loss1)
                print("value loss")
                print(loss2)
                print("total loss")
                '''
                print((loss1.item(),loss2.item()))
                loss.backward()
                optimizer.step()


    '''
    processes = []

    for i in range(4):
        print("hi")
        generator = data_generator.DataGenerator(network, rollouts, replaybuffer)
        process = mp.Process(target = generator.generate_games)
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
    '''

    #pool = mp.Pool(mp.cpu_count())
    #queue = mp.Queue(4)
    #datagenerators = [data_generator.DataGenerator(network, rollouts, replaybuffer) for i in range(4)]

    #for datagenerator in datagenerators:
        #queue.put(datagenerator.generate_games())
    #pool.map(datagenerator.generate_games(),  range(6))
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
        '''
        #input = torch.cat((
        #print(list(zip(game_data, value_data, policy_data)))
if __name__ == '__main__': main()
