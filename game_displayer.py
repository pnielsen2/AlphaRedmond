import pygame
import pygame.gfxdraw

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
        pygame.display.set_caption('AlphaRedmond')


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
        for intersection in black_intersections:
            drawstone(self.get_location(intersection), (0,0,0))
        for intersection in white_intersections:
            drawstone(self.get_location(intersection), (255,255,255))
        pygame.display.update()
