# 20 Number Challenge - Neural Net

## Purpose: Train a deep learning model to play the 20 numbers challenge game leveraging reinforcement learning techniques

## The Game
The 20 numbers challenge is game by Jonathan Hunter which can be found here: jonathanhunter.itch.io/20num
The game is simple; the game will generate a number from 1-999 and you must place it in 1 of 20 positions. Each number placed must be lower than every number in every higher position, and higher then every number in every lower position. You go as long as you can until you can no longer place the generated number, or you have placed all 20 numbers.

## The Code
The game is simulated in the file game_simulation.py
The model is in 20_number_challenge_nn.ipynb
To play the game simulation run the game_simulation.py
To train the model, the file game_simulation.py must be in the working directory.

## Results
![alt text](image.png)

## Model Description

The model is a 6 layer neural net with 21 input neurons (the current board state and the generated number), a 64 neuron layer, 2 128 neuron layers, a 64 neuron layer, and a 20 neuron output layer (that correspond to the 20 possible board position.) The model uses leaky ReLU, Adam optimization, and epsilon greedy Q-learning. In addition rewards are based on the average of 4 games with a replay buffer. I'll discuss these choices in the next section

## Discussion and Learnings

### Model Choices

#### Why leaky ReLU? 
As the model was training in earlier runs the model would increase steadily in average score, then suddenly and precipitously drop in performance and take ~10x as many epochs to recover somewhat until another precipitous drop would occur. I hypothesized that this was because the model was overfitting to some conditions that were not generalizable and would recover slowely because of neuron death. Switching to leaky ReLU mitigated the neuron death problem, reducing the frequency and severity of performance drops.

#### Why Adam optimization?

Adam optimization was faster and tested similarly in smaller runs to other optimizers. As this model was being run on limited compute resources, Adam seemed like the natural choice.

#### Why Q-learning?

Originally I tried a model without q-learning, but results were very bad. The model would frequently get stuck on average scores of under 4. The model was inappropriately minimizing loss towards an "optimal" play which was not provided in a sensical way. Q-learning was an effective method to train the model to maximize scores in this game environment and train the model to maximize the scores without providing "correct" answers.

#### Why Epsilon Greedy?

The model is very apt to plateau in performance. Runs without random actions would often not improve even over many epochs. I also tried the model with epsilon resetting, introducing more random actions periodically during the training, but found that the model immediately, consistently, and precipitously decreased in performance every time epsilon reset, even to "low" values like 0.1. 

#### Why Average Games?

I wanted some consistancy in performance. The idea was to not randomly generate a perfect game in enough tries, but to have a model that plays at roughly human level. I was aiming for an average of 10 points per game with infrequent scores above 17 - roughly matching my own experience in the hour or so I tried the game.

#### Why Replay Buffer?

The problems with sudden precipitous performance decline and difficulty recoving was not completely solved with leaky ReLU. The replay buffer allows the model to update on the experience of many past games at once, reducing variation and the chances of learning very bad strategies from a few sequential bad games.

#### Additional Choice - Masking Invalid Moves

Invalid moves stumped me for a while. If given the opportunity the model will make many invalid moves. I tried punishing the model for invalid moves, but even a small punishment would completely overwhelm the score and the model would not learn. Also games with higher scores have more invalid moves - as more positions are taken so less moves are possible. Thus the model was being punished in higher scoring games disproportionately, leading to pervurse results. 

The only option I could think of to get around punishment is to just mask the invalid moves. I don't know if this is optimal as part of the model's output is being ignored every round, which seems like it should lead to unintended consequences. I think a lot of the model's performance issues and inability to get to a 10 point average could be traced to the masking. I think enough epochs might aleviate the problem but I can't be sure without more compute resources than I willing to give this.

### General Learnings

This is my second deep learning project and my first time training a model on a game. Everything above reflects genuinely new material and learnings for me. Here is a bulleted list of observations, most of which will make me sound like a child:

    - The reward function is everything
    - A simple game can be hard to learn
    - Sometimes it would be a lot easier to program something to play heuristically
    - A model that suddenly gets bad gives more information than a model that is ok but plateaued
    - Getting a score of 20 on this game would be both impressive and random

### What's next?

For now this project is on the shelf. I don't know how good a model on this game can be and have concerns on that front. Before I work on this again I will work another game simulated in python, but with less randomness (maybe solitair?) That experience will eliminate some of the variables with this project and let me try some new techniques with more direct feedback. Once that is done I'll start from scratch on this again and try to get that average of 10.
