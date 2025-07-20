import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

csv_path = os.path.join(os.path.dirname(__file__), 'data.csv')
df = pd.read_csv(csv_path)

df_success = df[df['status'] == 'success']

grouped = df_success.groupby('iteration')

max_power = grouped['power_output'].max()
mean_power = grouped['power_output'].mean()

plt.figure(figsize=(10, 6))
plt.plot(np.asarray(max_power.index), np.asarray(max_power.values), label='Max Power per Iteration', marker='o')
plt.plot(np.asarray(mean_power.index), np.asarray(mean_power.values), label='Mean Power per Iteration', marker='x', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('Power Output')
plt.title('Model Training Progress (Power Output per Iteration)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(os.path.dirname(__file__), 'training_progress.png'))
plt.show()
