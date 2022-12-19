import matplotlib.pyplot as plt
import time

class LiveUpdater:
    def __init__(self):
        # Create a figure and two subplots
        self.fig, (self.loss_axis, self.duration_axis) = plt.subplots(2, 1)

        # Initialize empty lists to store the loss and duration data
        self.loss_data = []
        self.duration_data = []

        # Set the x-axis data to be the indices of the data points
        self.x_data = list(range(len(self.loss_data)))

    def update_plots(self, loss, duration):
        # Add the new loss and duration data to the lists
        self.loss_data.append(loss)
        self.duration_data.append(duration)

        # Update the x-axis data to be the indices of the data points
        self.x_data = list(range(len(self.loss_data)))

        # Clear the previous plots
        self.loss_axis.clear()
        self.duration_axis.clear()

        # Plot the new loss data
        self.loss_axis.plot(self.x_data, self.loss_data)

        # Plot the new duration data
        self.duration_axis.plot(self.x_data, self.duration_data)

        # Redraw the figure
        self.fig.canvas.draw()
        plt.pause(0.01)

updater = LiveUpdater()

for i in range(100):
    loss = i * i
    duration = i / 2
    updater.update_plots(loss, duration)
    time.sleep(0.5)



