import torch
import random
import numpy as np
from collections import deque
import math
from game import *
from model import *
from helper import *

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # the learning rate

class Agent:

    def __init__(self):

        self.num_games = 0
        self.epsilon = 0 # the randomness
        self.gamma = 0.9 # the discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # there are 11 parameters for input state, and 3 parameters for output
        self.model = Linear_QNet(11, 128, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):

        # get the position of the snake's head
        head = game.snake[0]

        # get the coordinates of four points surrounding the head of the snake
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # get the current direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # there are 11 parameters
        # [danger ahead, danger right, danger left, direction left, direction right, direction up, direction down
        # food left, food right, food up, food down]
        state = [

            # whether there is a danger ahead
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # whether there is a danger on the right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # whether there is a danger on the left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # the directions
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # the location of the food
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]

        # convert the data type to numpy
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):

        # append the info as a tuple to the memory deque
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):

        "train the model based on a batch of memory points (for pytorch)"

        # check if we have enough data points, if we do then we randomly pick BATCH_SIZE from our memory
        # if not, then we use all data points
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns a list of tuples
        else:
            mini_sample = self.memory

        # put all states, actions, etc. together
        states, actions, rewards, next_states, dones = zip(*mini_sample)    # unzip the list of tuples
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):

        # train based on a single sample
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):

        "get the action of the next move"

        # exploration/exploitation part
        self.epsilon = (0.8*math.exp(-0.01*self.num_games))
        final_move = [0,0,0]
        move = 0

        if random.random() < self.epsilon:

            # make a random move
            move = random.randint(0, 2) # returns an int between 0 & 2

        else:

            # make a move based on our model
            state0 = torch.tensor(state, dtype=torch.float) # convert the state to tensor data type
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()  # pick the highest score and get its index

        final_move[move] = 1
        return final_move


def train():

    "the main function"

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0  # the best score

    # instantiate classes
    agent = Agent()
    game = SnakeGameAI()

    # the main loop
    while True:

        # get the current state (old state)
        state_old = agent.get_state(game)

        # get the action
        final_move = agent.get_action(state_old)

       # perform based on the action
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train the short-term memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # store the memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:

            # reset the environment to default
            game.reset()

            # increase the number of games by one
            agent.num_games += 1

            # train the long-term memory
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            # print the reuslts
            print('Game', agent.num_games, 'Score', score, 'Record:', record)

            # plot the reuslts
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':

    train()
