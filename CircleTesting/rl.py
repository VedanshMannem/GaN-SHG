import numpy as np
import csv
import os
from testingcircle import runSim1  # Make sure runSim1 accepts a vector of params

param_bounds = {
    "radius": (1.064e-6, 2.0e-6),
    "GaNzSpan": (1.064e-6, 2.128e-6),
    "SapSpan": (1.064e-6, 5.0e-6),
    "theta": (0, 85),  # degrees
}

param_names = list(param_bounds.keys())
n_params = len(param_names)

# Initialize means at the middle of bounds
mean = np.array([(low + high) / 2 for (low, high) in param_bounds.values()])
sigma = 0.1 * mean  # Initial standard deviation (10% of mean)

population_size = 10
learning_rate = 0.2
n_iterations = 30

# Initialize CSV file for logging
csv_filename = "../CircleTesting/rl_data.csv"
csv_headers = ["iteration", "sample"] + param_names + ["power_output", "status"]

# Create CSV file and write headers
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_headers)

print(f"Created {csv_filename} for logging simulation results")

for iteration in range(n_iterations):
    print(f"\n=== Iteration {iteration+1}/{n_iterations} ===")
    noise = np.random.randn(population_size, n_params) * sigma
    samples = mean + noise

    # Clip to valid parameter ranges
    for i, name in enumerate(param_names):
        low, high = param_bounds[name]
        samples[:, i] = np.clip(samples[:, i], low, high)

    rewards = []
    for i, p in enumerate(samples):
        try:
            # Unpack the parameter vector to individual parameters
            radius, GaNzSpan, SapSpan, theta = p
            print(f"  Sample {i+1}: radius={radius:.2e}, GaNzSpan={GaNzSpan:.2e}, SapSpan={SapSpan:.2e}, theta={theta:.1f}")
            
            power = runSim1(radius, GaNzSpan, SapSpan, theta)
            
            # Extract scalar value if it's an array
            if isinstance(power, np.ndarray):
                power_value = float(power.item())
            else:
                power_value = float(power)
                
            print(f"    Power result: {power_value}")
            rewards.append(power_value)
            
            # Log successful simulation to CSV
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                row = [iteration+1, i+1, radius, GaNzSpan, SapSpan, theta, power_value, "success"]
                writer.writerow(row)
                
        except Exception as e:
            print(f"Simulation failed for sample {i+1}: {p}. Error: {e}")
            rewards.append(0)
            
            # Log failed simulation to CSV
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                row = [iteration+1, i+1, p[0], p[1], p[2], p[3], 0.0, f"failed: {str(e)[:50]}"]
                writer.writerow(row)

    rewards = np.array(rewards)

    # Normalize rewards
    if np.std(rewards) > 0:
        rewards = (rewards - np.mean(rewards)) / np.std(rewards)

    # Update mean (gradient ascent)
    gradient_estimate = np.dot(noise.T, rewards) / population_size
    mean += learning_rate * gradient_estimate

    # Clip mean to stay valid
    for i, name in enumerate(param_names):
        low, high = param_bounds[name]
        mean[i] = np.clip(mean[i], low, high)

    print(f"Iteration {iteration+1}: Current Best Params:")
    for i, name in enumerate(param_names):
        print(f"  {name}: {mean[i]}")

print("\nOptimized Parameters:")
for i, name in enumerate(param_names):
    print(f"{name}: {mean[i]}")
