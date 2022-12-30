# AI Snake Project

## Overview

We use Deep Q-Learning method to train our agent to play Snake game.

## Approaches

## Results

<figure>
  <img src="https://github.com/neilchen1998/ai-snake/blob/main/gifs/training-early-stage.gif" alt="my alt text" width="300" height="250"/>
  <figcaption align="bottom">The early stage of training</figcaption>
</figure>

<figure>
  <img src="https://github.com/neilchen1998/ai-snake/blob/main/gifs/training-late-stage.gif" alt="my alt text" width="300" height="250"/>
  <figcaption align="bottom">After 350 episodes of training</figcaption>
</figure>

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
