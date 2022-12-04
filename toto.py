
import matplotlib.pyplot as plt    
import numpy as np

#plot a relu function
def plot_relu():
    x = np.linspace(-3,3,100)
    y = np.maximum(x,0)
    plt.plot(x,y, label='relu')
    plt.legend()
    plt.show()

#plot relu with a 0.5 shift to the right   
def plot_relu_shift():
    x = np.linspace(-10,10,100)
    y = np.maximum(x-0.5,0)
    plt.plot(x,y)
    plt.show()


#plot relu with a threshold of 0.5
def plot_relu_threshold():
    x = np.linspace(-3,3,10000)
    y2 = np.where(x<0.5, 0, x)
    #add series labels
    #plt.plot(x,y, label='relu')
    plt.plot(x,y2, label='relu with 0.5 threshold')
    #add legend
    plt.legend()
    #add threshold line
    #plt.axvline(x=0.5, color='r', linestyle='--')
    plt.show()


plot_relu()