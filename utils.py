
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
5) saves the scores, episode durations and loss in a json file
"""
def save_model(policy_model, target_model, agent, options, scores, durations, loss, root="checkpoints", save_memory=False):
    # create a folder with the current time in its name
    folder_name = time.strftime("{}/%Y%m%d-%H%M%S".format(root))
    os.mkdir(folder_name)
    # save the torch model in this folder
    torch.save(policy_model.state_dict(), folder_name + "/policy_model.pt")
    torch.save(target_model.state_dict(), folder_name + "/target_model.pt")
    # save the model class (only the name of the class)
    with open(folder_name + "/model_class.json", "w") as f:
        data = {"policy_model": policy_model.__class__.__name__, "target_model": target_model.__class__.__name__}
        json.dump(data, f)
    # save the parameters used to train the model
    with open(folder_name + "/trainer_params.json", "w") as f:
        json.dump(agent.get_params(), f)
    # save the parameters used to create the environment
    with open(folder_name + "/env_params.json", "w") as f:
        json.dump(options.get_params(), f)
    # save the scores, episode durations and loss in a json file
    with open(folder_name + "/scores.json", "w") as f:
        json.dump(scores, f)
    with open(folder_name + "/durations.json", "w") as f:
        json.dump(durations, f)
    with open(folder_name + "/loss.json", "w") as f:
        json.dump(loss, f)
    #save agent memory
    if save_memory:
        with open(folder_name + "/agent_memory.json", "w") as f:
            json.dump(agent.memory, f)
    return folder_name

"""
method that for a given save folder returns the weights, the model class and the environment poarameters
"""
def load_model(folder_name):
    # load the model class
    with open(folder_name + "/model_class.json", "r") as f:
        data = json.load(f)
    policy_model_class = data["policy_model"]
    target_model_class = data["target_model"]
    # load the model
    policy_weights = torch.load(folder_name + "/policy_model.pt")
    target_weights = torch.load(folder_name + "/target_model.pt")
    # load the environment parameters
    with open(folder_name + "/env_params.json", "r") as f:
        options = json.load(f)

    return policy_weights, target_weights, policy_model_class, target_model_class, options

def load_memory(folder_name):
    with open(folder_name + "/agent_memory.json", "r") as f:
        memory = json.load(f)
    return memory

"""
method that for a given save folder plots the scores, episode durations and loss in a composite plot
"""
def plot_model_training_history(folder_name):
    # load the scores, episode durations and loss
    with open(folder_name + "/scores.json", "r") as f:
        scores = json.load(f)
    with open(folder_name + "/durations.json", "r") as f:
        durations = json.load(f)
    with open(folder_name + "/loss.json", "r") as f:
        loss = json.load(f)
    # plot the scores, episode durations and loss
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(scores)
    axes[0].set_title("Score per episode")
    axes[1].plot(durations)
    axes[1].set_title("Duration per episode")
    axes[2].plot(loss)
    axes[2].set_title("Loss per batch")
    plt.show()