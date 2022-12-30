# AI Snake Project

## Overview

We use Deep Q-Learning method to train our agent to play Snake game.

## Approaches

Firstly, we defined the state as a list of 11 boolean parameters: [danger ahead, danger right, danger left, direction left, direction right, direction up, direction down, food left, food right, food up, food down]. The agent will only have access to those 11 parameters.

We define the learning rate as $\alpha$, the randomness as $\epsilon$, and the discount rate as $\gamma$.

We use PyTorch libraries to build our deep Q-learning model.
We take the advantage of experience replay, which the agent learns from a batch of experience (choose 1,000 samples randomly from the experience of 100,000 data points). The agent then makes an action: a random action (exploration) or an action based on the agent's experience (exploitation) based on the value of a random number that it generates.

We only train the agent for 450 episodes and we record the total avgerage score, the last 100 episode average score, and the best score during the training period.

## Results

<figure>
  <img src="https://github.com/neilchen1998/ai-snake/blob/main/gifs/training-early-stage.gif" alt="my alt text" width="300" height="250"/>
  <figcaption align="bottom">The early stage of training</figcaption>
</figure>

<figure>
  <img src="https://github.com/neilchen1998/ai-snake/blob/main/gifs/training-late-stage.gif" alt="my alt text" width="300" height="250"/>
  <figcaption align="bottom">After 350 episodes of training</figcaption>
</figure>

<figure>
  <img src="https://github.com/neilchen1998/ai-snake/blob/main/gifs/result-graph.png" alt="my alt text" width="300" height="250"/>
  <figcaption align="bottom">A snapshot of the result graph</figcaption>
</figure>

&nbsp;

<table>
<caption align="center">Results</caption>
<tr>
    <th>Model</th>
    <th>Gamma</th>
    <th>Food Rwrd.</th>
    <th>Collide Rwrd.</th>
    <th>TOT Avg. Score</th>
    <th>Last 100 Avg. Score</th>
    <th>Best Score</th>
</tr>
<tr>
  <td rowspan="4">three layers</td>
  <td rowspan="2">0.9</td>
  <td>20</td>
  <td>-10</td>
  <td>9.653</td>
  <td>22.9</td>
  <td>59</td>
</tr>
<tr>
  <td>20</td>
  <td>-50</td>
  <td>7.04</td>
  <td>21.61</td>
  <td>52</td>
</tr>
<tr>
  <td>0.85</td>
  <td>20</td>
  <td>-10</td>
  <td>9.412</td>
  <td>23.56</td>
  <td>57</td>
</tr>
<tr>
  <td>0.95</td>
  <td>20</td>
  <td>-10</td>
  <td>9.507</td>
  <td>25.44</td>
  <td>63</td>
</tr>
<tr>
  <td>two layers (128 nodes)</td>
  <td>0.9</td>
  <td>20</td>
  <td>-10</td>
  <td>9.551</td>
  <td>26.22</td>
  <td>52</td>
</tr>
<tr>
  <td>two layers (256 nodes)</td>
  <td>0.9</td>
  <td>20</td>
  <td>-10</td>
  <td>10.704</td>
  <td>24.64</td>
  <td>55</td>
 </tr>
</table>

## References

1. [Deep Reinforcement Learning: Guide to Deep Q-Learning](https://www.mlq.ai/deep-reinforcement-learning-q-learning/)

2. [Building Model with PyTorch](https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)
