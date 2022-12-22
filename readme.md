Pygame with reinforcement learning

Install using coda or venv
Then `pip install -t requirements.txt`

Things one can do:

- Run the game and play it: `python main.py`. Use arrows to move, ESC to quit
- Train a model using `training_pipe.py`
- Playback a trained model using `model_player.py` (just give the code the folder path of the trained model)

Code documentation

- game.py contains the game (or enviroment), written using Pygame framework and compliant to openAI's Gym environment interface
   - the gym interfaces allows hooking an agent (AI) to control it. It provides the observation space (what the agent sees) and the action (what the agent can do) and the methods to reset and step through the game. It supports different ways of rendering the game depending on what the agent (or the developer) needs.
   - the game supports different options to control game parameters (screen size, complexity of the game)
   - sprites.py defines the different objects used in the game
- agent.py contains an implementation of a DQN agent. The agent is tested against an existing game (cartpole) in cartpole_test.py. The agent implementation comes from a Pytorch tutorial. The agent can show live plots as it trains or the final plot. The agent can also find success condition and do early stopping. The agent has a number of hyperparameters to control the algorithm
- models.py contains the NN models to use with the agent.
- training_pipe.py brings all together by creating the game, the model and the agent. It can iterate through parameter combinations to find the best combination. 
- utils.py provides save/load capabilities 