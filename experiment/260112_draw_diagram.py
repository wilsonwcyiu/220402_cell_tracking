import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ----------------------
# Data declaration
# ----------------------
np.random.seed(42)

# Numeric x-axis data
x = np.arange(1, 6)
y_linear = x
y_quadratic = x**2

# Data for boxplot
df = pd.DataFrame({
    "Group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
    "Value": np.concatenate([
        np.random.normal(5, 1, 20),
        np.random.normal(7, 1.5, 20),
        np.random.normal(6, 1, 20)
    ])
})

# ----------------------
# Create subplots
# ----------------------
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(5.12, 5.12),  # 512 x 512 pixels @ dpi=100
    dpi=100
)

# ----------------------
# Top-left: Line plot
# ----------------------
axes[0, 0].plot(x, y_linear, marker='o')
axes[0, 0].set_title("Linear")

# ----------------------
# Top-right: Quadratic plot
# ----------------------
axes[0, 1].plot(x, y_quadratic, marker='o')
axes[0, 1].set_title("Quadratic")

# ----------------------
# Bottom-left: Scatter plot
# ----------------------
axes[1, 0].scatter(x, y_quadratic)
axes[1, 0].set_title("Scatter")

# ----------------------
# Bottom-right: Seaborn boxplot
# ----------------------
sns.boxplot(
    data=df,
    x="Group",
    y="Value",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Seaborn Boxplot")

# ----------------------
# Shared formatting
# ----------------------
for ax in axes.flat:
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

plt.tight_layout()
plt.show()