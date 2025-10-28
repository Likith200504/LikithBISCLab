import numpy as np
import time

# Parameters
road_length = 20
time_steps = 10

# Initialize road: 0 = empty, 1 = car
road = np.zeros(road_length, dtype=int)
road[[2, 5, 8]] = 1  # place cars

print("Initial Road:")
print(road)

# Simulation loop
for t in range(time_steps):
    new_road = np.zeros(road_length, dtype=int)
    for i in range(road_length):
        if road[i] == 1:
            next_cell = (i + 1) % road_length
            if road[next_cell] == 0:
                new_road[next_cell] = 1  # move car forward
            else:
                new_road[i] = 1  # stay in place
    road = new_road
    print(f"Time step {t+1}:")
    print(road)
    time.sleep(0.5)
