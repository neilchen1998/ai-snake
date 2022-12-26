import os
import torch
import random
import numpy as np
from collections import deque
import math
import csv
from game import *
from model import *
from helper import *

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # the learning rate

class Agent:

    def __init__(self):

        self.episodes = 0
        self.epsilon = 0 # the randomness
        self.gamma = 0.9 # the discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # there are 11 parameters for input state, and 3 parameters for output
        self.model = Linear_QNet(11, 128, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.log_file_name = "{}-{}-{}.csv".format(self.gamma, "three-layers", "250")

        # check if the file already exists (to prevent from overwriting)
        self._check_file_exist()

    def _check_file_exist(self):

        flag = os.path.exists(self.log_file_name)

        if flag:
            raise AssertionError("File already exists")

    def get_state(self, game):

        # get the position of the snake's head
        head = game.snake[0]

        # get the coordinates of four points surrounding the head of the snake
        pt_left = Point(head.x - 20, head.y)
        pt_right = Point(head.x + 20, head.y)
        pt_up = Point(head.x, head.y - 20)
        pt_down = Point(head.x, head.y + 20)

        # get the current direction
        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        # there are 11 parameters
        # [danger ahead, danger right, danger left, direction left, direction right, direction up, direction down
        # food left, food right, food up, food down]
        state = [

            # whether there is a danger ahead
            (dir_right and game.collide(pt_right)) or
            (dir_left and game.collide(pt_left)) or
            (dir_up and game.collide(pt_up)) or
            (dir_down and game.collide(pt_down)),

            # whether there is a danger on the right
            (dir_up and game.collide(pt_right)) or
            (dir_down and game.collide(pt_left)) or
            (dir_left and game.collide(pt_up)) or
            (dir_right and game.collide(pt_down)),

            # whether there is a danger on the left
            (dir_down and game.collide(pt_right)) or
            (dir_up and game.collide(pt_left)) or
            (dir_right and game.collide(pt_up)) or
            (dir_left and game.collide(pt_down)),

            # the directions
            dir_left,
            dir_right,
            dir_up,
            dir_down,

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
            samples = random.sample(self.memory, BATCH_SIZE) # returns a list of tuples
        else:
            samples = self.memory

        # put all states, actions, etc. together
        states, actions, rewards, next_states, dones = zip(*samples)    # unzip the list of tuples
        self.trainer.training_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):

        # train based on a single sample
        self.trainer.training_step(state, action, reward, next_state, done)

    def get_action(self, state):

        "get the action of the next move"

        # exploration/exploitation part
        self.epsilon = (0.8*math.exp(-0.01*self.episodes))
        next_move = [0,0,0]
        move_dir = 0

        if random.random() < self.epsilon:

            # make a random move
            move_dir = random.randint(0, 2) # returns an int between 0 & 2

        else:

            # make a move based on our model
            state0 = torch.tensor(state, dtype=torch.float) # convert the state to tensor data type
            prediction = self.model(state0)
            move_dir = torch.argmax(prediction).item()  # pick the highest score and get its index

        next_move[move_dir] = 1
        return next_move


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
    while (agent.episodes < 450):

        # get the current state (old state)
        state_old = agent.get_state(game)

        # get the action
        final_move = agent.get_action(state_old)

       # perform based on the action
        reward, done, score, num_steps = game.next_step(final_move)
        state_new = agent.get_state(game)

        # train the short-term memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # store the memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:

            # reset the environment to default
            game.reset()

            # increase the number of games by one
            agent.episodes += 1

            # train the long-term memory
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save_model()

            # print the reuslts
            print('Episode:', agent.episodes, 'Score:', score, 'Steps:', num_steps, 'Best:', record)

            # output the results to log file
            with open(agent.log_file_name, 'a', newline='') as file:  # use append method
            
                writer = csv.writer(file)
                data = [agent.episodes, score, num_steps]
                writer.writerow(data)

            # plot the reuslts
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.episodes
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':

    train()
