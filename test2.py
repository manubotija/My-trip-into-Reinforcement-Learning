import pandas as pd

# Define the hyperParams object
class HyperParams:
    def __init__(self, gamma, eps_decay, eps_end, batch_size):
        self.gamma = gamma
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.batch_size = batch_size

def _init_table():
    # Create a DataFrame with column names specified
    table = pd.DataFrame(columns=['weights_path','gamma', 'eps_decay', 'eps_end', 'batch_size', 'reward_type', 'average_score', 'speed', 'success'])
    return table

def _add_to_table(table, hyperParams, average_score, speed, weights_path, reward_type, success):
    # Add a new row to the dataframe
    table = table.append({'weights_path': weights_path,
                        'gamma': hyperParams.gamma,
                        'eps_decay': hyperParams.eps_decay,
                        'eps_end': hyperParams.eps_end,
                        'batch_size': hyperParams.batch_size,
                        'reward_type': reward_type,
                        'average_score': average_score,
                        'speed': speed, 
                        'success': success
                        }, ignore_index=True)
    return table

# Initialize the hyperParams object
hyperParams = HyperParams(0.9, 0.995, 0.01, 32)

# Initialize the table
table = _init_table()

# Add a new row to the table
table = _add_to_table(table, hyperParams, 10, 10, "aaaa", 1, True)

# Print the table
print(table)





