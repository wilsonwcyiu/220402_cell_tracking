import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Sample data
group1 = np.array([23, 25, 27, 29, 30, 28, 26])
group2 = np.array([31, 33, 35, 34, 36, 32, 34])

# Perform t-test
t_stat, p_value = ttest_ind(group1, group2)

# Calculate means and standard deviations
means = [group1.mean(), group2.mean()]
stds = [group1.std(ddof=1), group2.std(ddof=1)]

# Plot
plt.figure()
plt.bar(['Group 1', 'Group 2'], means, yerr=stds, capsize=5)
plt.ylabel('Value')
plt.title(f'T-test Result\n t = {t_stat:.2f}, p = {p_value:.4f}')

plt.show()