import matplotlib.pyplot as plt
import numpy as np


# List of (x, y) coordinates
coord_tuple_list = [(1, 2), (2, 4), (3, 1), (4, 3), (5, 2)]


# Split into x and y
x = [p[0] for p in coord_tuple_list]
y = [p[1] for p in coord_tuple_list]

# Plot coord_tuple_list + lines
fig = plt.figure(figsize=(51.2, 51.2), dpi=100)
# plt.figure()
plt.plot(x, y, marker='o', linewidth=2)  # line + coord_tuple_list

# Labels
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Diagram with Connected Lines")

sequence = np.arange(len(coord_tuple_list))
scatter = plt.scatter(
    x, y,
    c=sequence,
    cmap='viridis',   # perceptually uniform
    s=60
)

cmap='viridis_r'

for i in range(len(x) - 1):
    plt.arrow(
        x[i], y[i],
        x[i+1] - x[i], y[i+1] - y[i],
        length_includes_head=True,
        head_width=0.05,
        alpha=0.6
    )

plt.show()
