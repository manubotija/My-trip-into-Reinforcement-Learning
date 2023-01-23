
I launched into learning Pygame with reinforcement learning.

I have documented my whole journey while learning here:
https://medium.com/@manubotija/list/my-trip-into-reinforcement-learning-d6c244d5aa29 

# Install using coda or venv
`pip install -r requirements.txt`

Since using specific branch of sb3 that supports gym 0.26, install using following command:

`pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests`

see https://github.com/DLR-RM/stable-baselines3/pull/780 for more information.


# Running

Running `python main.py -h` provides all options available. Some examples are

Play a game:

```bash
python main.py play mid-barrier-no-proj config-4
```

Train a model:

```bash
python main.py train mid-barrier-no-proj config-4 --time_steps 300000 --project_name TEST
```

Evaluate a model:

```bash
python main.py evaluate mid-barrier-no-proj config-4 --model_path path/to/best_model.zip --render
```

TODO: 
- Run hyperparam search from CLI



# Old code


Under `old_code` is my the implementation of DQN, various utilities and training pipeline (up to [DAY 12](https://medium.com/@manubotija/day-12-my-trip-to-reinforcement-learning-9564500a1379)). None of it probably works out of the box since I have not kept the game nor the wrapper backwards compatible 