import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Setting a seed for reproducibility
np.random.seed(42)

# Creating DataFrames for PPO and SAC experiments
data_ppo = {
    'Run1': np.random.normal(2, 0.5, 50),
    'Run2': np.random.normal(2, 0.7, 50),
    'Run3': np.random.normal(2, 0.4, 50),
    'Run4': np.random.normal(2, 0.6, 50),
    'Run5': np.random.normal(2, 0.5, 50),
    'Run6': np.random.normal(2, 0.4, 50),
    'Run7': np.random.normal(2, 0.6, 50),
    'Run8': np.random.normal(2, 0.5, 50),
    'Run9': np.random.normal(2, 0.5, 50),
    'Run10': np.random.normal(2, 0.7, 50),
    'Run11': np.random.normal(2, 0.4, 50),
    'Run12': np.random.normal(2, 0.6, 50),
    'Run13': np.random.normal(2, 0.5, 50),
    'Run14': np.random.normal(2, 0.4, 50),
    'Run15': np.random.normal(2, 0.6, 50),
    'Run16': np.random.normal(2, 0.5, 50)
}
data_sac = {
    'Run1': np.random.normal(2.5, 0.5, 50),
    'Run2': np.random.normal(2.5, 0.7, 50),
    'Run3': np.random.normal(2.5, 0.4, 50),
    'Run4': np.random.normal(2.5, 0.6, 50),
    'Run5': np.random.normal(2.5, 0.5, 50)
}

# Convert to DataFrames
df_ppo = pd.DataFrame(data_ppo, index=range(1, 51))
df_sac = pd.DataFrame(data_sac, index=range(1, 51))

# Adding experiment identifiers
df_ppo['Experiment'] = 'PPO'
df_sac['Experiment'] = 'SAC'

# Combine into a single DataFrame
df_combined = pd.concat([df_ppo, df_sac], axis=0)

# Calculating rolling means for each run
rolling_window = 10
for column in df_combined.columns[:-1]:  # Exclude the 'Experiment' column
    df_combined[column] = df_combined[column].rolling(window=rolling_window, min_periods=1).mean()

# Melting the DataFrame to long format suitable for Seaborn
df_melted = df_combined.reset_index().melt(id_vars=['index', 'Experiment'], var_name='Run', value_name='Return')

# Plotting with Seaborn using the new 'errorbar' parameter (sd for stardard deviation and se for standard error)
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_melted, x='index', y='Return', hue='Experiment', style='Experiment', estimator='mean', errorbar=('se', 1))

# Ensuring x-axis ticks are integers and adjust automatically
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Smoothed Mean and Standard Error of Returns Across Episodes for PPO vs SAC', fontsize=16)
plt.xlabel('Episode', fontsize=14)
plt.ylabel('Return', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Experiment', title_fontsize='13', fontsize='12')
plt.grid(True)
plt.show()

