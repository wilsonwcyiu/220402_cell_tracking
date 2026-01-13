import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem

# Example data
df = pd.DataFrame({
    "value": [5, 6, 7, 6, 5, 6, 8, 9, 7, 8, 9, 8],
    "group": ["Group 1"] * 6 + ["Group 2"] * 6
})

plt.figure()

# Bar plot (set palette explicitly)
sns.barplot(
    data=df,
    x="group",
    y="value",
    palette=["skyblue", "lightgreen"],  # bar colors
    errorbar=lambda x: sem(x),
    alpha=0.7
)

# Strip plot (force a different color)
sns.stripplot(
    data=df,
    x="group",
    y="value",
    color="black",    # <-- explicitly different from bars
    size=6,
    jitter=True,
    zorder=10
)

plt.ylabel("Value")
plt.title("Bar plot with differently colored stripplot overlay")
plt.tight_layout()
plt.show()