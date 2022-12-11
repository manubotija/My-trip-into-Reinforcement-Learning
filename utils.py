
import torch
import time
import json
import os
import matplotlib.pyplot as plt
""" 
method that 
1) creates a new folder with the current time in its name
2) saves the torch model in this folder
3) saves a json file with the parameters used to train the model
4) saves a json file with the parameters used to create the environment
5) saves the rewards, episode durations and loss in a json file
"""
def save_model(model, trainer, options, rewards, durations, loss):
    # create a folder with the current time in its name
    folder_name = time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(folder_name)
    # save the torch model in this folder
    torch.save(model.state_dict(), folder_name + "/model.pt")
    # save the model class (only the name of the class)
    with open(folder_name + "/model_class.json", "w") as f:
        json.dump(model.__class__.__name__, f)
    # save the parameters used to train the model
    with open(folder_name + "/trainer_params.json", "w") as f:
        json.dump(trainer.get_params(), f)
    # save the parameters used to create the environment
    with open(folder_name + "/env_params.json", "w") as f:
        json.dump(options.get_params(), f)
    # save the rewards, episode durations and loss in a json file
    with open(folder_name + "/rewards.json", "w") as f:
        json.dump(rewards, f)
    with open(folder_name + "/durations.json", "w") as f:
        json.dump(durations, f)
    with open(folder_name + "/loss.json", "w") as f:
        json.dump(loss, f)
    return folder_name

"""
method that for a given save folder returns the weights, the model class and the environment poarameters
"""
def load_model(folder_name):
    # load the model class
    with open(folder_name + "/model_class.json", "r") as f:
        model_class = json.load(f)
    # load the model
    weights = torch.load(folder_name + "/model.pt")
    # load the environment parameters
    with open(folder_name + "/env_params.json", "r") as f:
        options = json.load(f)

    return model_class, weights, options

"""
method that for a given save folder plots the rewards, episode durations and loss in a composite plot
"""
def plot_model_training_history(folder_name):
    # load the rewards, episode durations and loss
    with open(folder_name + "/rewards.json", "r") as f:
        rewards = json.load(f)
    with open(folder_name + "/durations.json", "r") as f:
        durations = json.load(f)
    with open(folder_name + "/loss.json", "r") as f:
        loss = json.load(f)
    # plot the rewards, episode durations and loss
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(rewards)
    axes[0].set_title("Rewards per episode")
    axes[1].plot(durations)
    axes[1].set_title("Duration per episode")
    axes[2].plot(loss)
    axes[2].set_title("Loss per batch")
    plt.show()