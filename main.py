import pygame
from pygame.locals import *
import numpy as np
from multiprocessing import Process
import random

import time
import math
import copy

import game_displayer
import game_simulator
import agents
import networks

#test




def main():


    # display parameters
    display = True
    windowwidth = 1030
    windowheight = 830
    boardsize = 810

    # game parameters
    dimension = 9
    number_of_games = 10



    squaresize = boardsize / dimension
    margin = (min([windowwidth, windowheight]) - boardsize) / 2

    # nontrivial initializations (will change in distant future
    displayer = None if display == False else game_displayer.GameWindow(windowwidth, windowheight, boardsize, dimension)



    for i in range(number_of_games):
        network = networks.Network()
        [black_agent, white_agent] = [agents.NNAgent(network), agents.NNAgent(network)]
        gamesim = game_simulator.GameSim(0, dimension)

        just_was_clicking = False
        run = True

        gamesim.run(displayer, [black_agent, white_agent])

if __name__ == '__main__': main()
