import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# class syntax
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# define a new tuple subclass for our coordiantes
Point = namedtuple('Point', 'x, y')

# colours
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

# some constants
BLOCK_SIZE = 20
SPEED = 100

class SnakeGameAI:

    def __init__(self, w=640, h=480):

        self.w = w
        self.h = h

        # initialize the game window
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # reset everything
        self.reset()


    def reset(self):

        # initialize the snake
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)   # we initialize the position of the snake's head at the center of the window
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        # initialize the game status
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):

        # we pick a random spot to place the food
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)

        # we want to make sure we do not want the food to be placed on the snake's body
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):

        # increase the time by one
        self.frame_iteration += 1

        # get the user's input
        for event in pygame.event.get():

            # check if user presses ESC
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # move the snake
        self._move(action) # update the position of the snake
        self.snake.insert(0, self.head) # insert the position of the new snake head to the front of the snake

        # check if the game is over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 250*len(self.snake):   # check for time out (the time length is proportional to the length of the snake)

            # if the snake collides with the wall or bites itself,
            # then the game is over and we return the score
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # check if the snake eats the food
        if self.head == self.food:

            # increase the score by one and place another food
            self.score += 1
            reward = 20
            self._place_food()

        # if the snake does not eat the food,
        # then we pop the tail so the length remains the same as before
        else:
            self.snake.pop()

        # update the frame
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score


    def is_collision(self, pt=None):

        "check if the snake hits the walls or eats itself"

        if pt is None:
            pt = self.head

        # check if the snake collides with the walls
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # check if the snake bites itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):

        "display the gaming environment"

        # fill the background as black
        self.display.fill(BLACK)

        # draw the body of the snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # draw the food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()   # update only a portion of the screen


    def _move(self, action):

        "move the snake based on the action received"

        # action = [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # calculate the new direction
        if action[0] == 1:

            # straight
            new_dir = clock_wise[idx]
        elif action[1] == 1:

            # turn right
            new_dir = clock_wise[(idx + 1) % 4]
        else:

            # turn left
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        # calculate the new position of the snake's head
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
