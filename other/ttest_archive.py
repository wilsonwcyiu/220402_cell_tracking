import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# -----------------------------
# Example data (replace with yours)
# -----------------------------
group1 = np.array([23, 25, 27, 21, 24])
group2 = np.array([30, 29, 31, 28, 32])

# -----------------------------
# Perform Welch's t-test
# -----------------------------
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

# -----------------------------
# Calculate statistics
# -----------------------------
means = [np.mean(group1), np.mean(group2)]
sems = [
    np.std(group1, ddof=1) / np.sqrt(len(group1)),
    np.std(group2, ddof=1) / np.sqrt(len(group2))
]

# -----------------------------
# Plot bar chart
# -----------------------------
plt.figure(figsize=(6, 5))

x_pos = np.arange(2)

# Bars (mean ± SEM)
plt.bar(x_pos, means, yerr=sems, capsize=6)

# Overlay raw data points
for i, group in enumerate([group1, group2]):
    jitter = np.random.normal(i, 0.05, size=len(group))
    plt.scatter(jitter, group)

# Labels
plt.xticks(x_pos, ['Group 1', 'Group 2'])
plt.ylabel('Value')
plt.title('Bar Chart with t-test')

# Annotate statistics
y_max = max(np.max(group1), np.max(group2))
plt.text(
    0.5,
    y_max * 1.05,
    f"t = {t_stat:.3f}\np = {p_value:.3e}",
    ha='center'
)

plt.tight_layout()
plt.show()
