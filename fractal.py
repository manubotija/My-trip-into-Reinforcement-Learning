import matplotlib.pyplot as plt
import random

# Define the vertices of the triangle
vertices = [(0.1, 0), (0.3, 0.866), (1, 0.3)]

# Define a starting point
point = (0.1, 0.1)

# Number of points to draw
num_points = 10000

# Array to store points
points = []

# Create points
for _ in range(num_points):
    # Append current point to the list
    points.append(point)
    
    # Randomly select a vertex
    vertex = random.choice(vertices)
    
    # Move half the distance from the current point to the selected vertex
    point = ((point[0] + vertex[0]) / 2, (point[1] + vertex[1]) / 2)

# Separate the x and y coordinates
x, y = zip(*points)

# Plot the points
plt.scatter(x, y, s=1, color='blue')
plt.show()
