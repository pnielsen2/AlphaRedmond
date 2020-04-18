class GameCoordinator():
    def __init__(displayer, agents, gamesim):
        self.displayer = displayer
        self.agents = agents
        self.gamesim = gamesim

    def run(self):
        while self.gamesim.game_over == False:
            next_move = self.get_action(displayer, agents[self.gamesim.current_player])

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
        print("hi")
        agent.ponder(self.gamesim)
        print("hi")
        while True:
            intersection = agent.get_intersection(illegal_moves)
            if intersection != None:
                legal = self.step(intersection)
                if legal:
                    break
            else:
                break
        return intersection
