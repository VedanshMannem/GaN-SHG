import numpy as np
import csv
import os
from triangle import runSim1  # Use runSim1 from triangle.py

# FDTD region is -1.5e-6 to +1.5e-6, keep 0.532e-6 away from edge
FDTD_LIMIT = 1.5e-6
MARGIN = 0.532e-6
COORD_MIN = -FDTD_LIMIT + MARGIN  # -0.968e-6
COORD_MAX = FDTD_LIMIT - MARGIN   # +0.968e-6

param_names = ["x1", "x2", "x3", "y1", "y2", "y3"]
n_params = 6

# Initial mean: equilateral triangle centered at (0,0)
mean = np.array([
    0.0,
    1e-6 * np.sqrt(3)/2,
    -1e-6 * np.sqrt(3)/2,
    -0.5e-6,
    0.5e-6,
    0.5e-6
])
sigma = 0.2 * (COORD_MAX - COORD_MIN) * np.ones(n_params)  # 20% of range

population_size = 10
learning_rate = 0.2
n_iterations = 30

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_filename = os.path.join(script_dir, "triRL_data.csv")
csv_headers = ["iteration", "sample"] + param_names + ["power_output", "status"]

with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_headers)

print(f"Created {csv_filename} for logging simulation results")

def triangle_area(x1, x2, x3, y1, y2, y3):
    return 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

MIN_AREA = 0.01e-12  # 0.01 um^2, adjust as needed
MIN_SIDE = 0.532e-6  # 0.532 microns in meters

def side_lengths(x1, x2, x3, y1, y2, y3):
    a = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    b = np.sqrt((x2-x3)**2 + (y2-y3)**2)
    c = np.sqrt((x3-x1)**2 + (y3-y1)**2)
    return a, b, c

for iteration in range(n_iterations):
    print(f"\n=== Iteration {iteration+1}/{n_iterations} ===")
    noise = np.random.randn(population_size, n_params) * sigma
    samples = mean + noise

    # Clip to valid parameter ranges
    samples = np.clip(samples, COORD_MIN, COORD_MAX)

    rewards = []
    for i, p in enumerate(samples):
        x1, x2, x3, y1, y2, y3 = p
        area = triangle_area(x1, x2, x3, y1, y2, y3)
        a, b, c = side_lengths(x1, x2, x3, y1, y2, y3)
        if area < MIN_AREA:
            print(f"  Sample {i+1}: Invalid triangle (area too small: {area:.2e})")
            rewards.append(0)
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                row = [iteration+1, i+1, x1, x2, x3, y1, y2, y3, 0.0, "invalid_triangle_area"]
                writer.writerow(row)
            continue
        if min(a, b, c) < MIN_SIDE:
            print(f"  Sample {i+1}: Invalid triangle (side too short: {min(a, b, c):.2e})")
            rewards.append(0)
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                row = [iteration+1, i+1, x1, x2, x3, y1, y2, y3, 0.0, "invalid_triangle_side"]
                writer.writerow(row)
            continue
        try:
            print(f"  Sample {i+1}: x1={x1:.2e}, x2={x2:.2e}, x3={x3:.2e}, y1={y1:.2e}, y2={y2:.2e}, y3={y3:.2e}, area={area:.2e}")
            power = runSim1(x1, x2, x3, y1, y2, y3)
            if isinstance(power, np.ndarray):
                power_value = float(power.item())
            else:
                power_value = float(power)
            print(f"    Power result: {power_value}")
            rewards.append(power_value)
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                row = [iteration+1, i+1, x1, x2, x3, y1, y2, y3, power_value, "success"]
                writer.writerow(row)
        except Exception as e:
            print(f"Simulation failed for sample {i+1}: {p}. Error: {e}")
            rewards.append(0)
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                row = [iteration+1, i+1, x1, x2, x3, y1, y2, y3, 0.0, f"failed: {str(e)[:50]}"]
                writer.writerow(row)

    rewards = np.array(rewards)
    if np.std(rewards) > 0:
        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
    gradient_estimate = np.dot(noise.T, rewards) / population_size
    mean += learning_rate * gradient_estimate
    mean = np.clip(mean, COORD_MIN, COORD_MAX)
    print(f"Iteration {iteration+1}: Current Best Params:")
    for j, name in enumerate(param_names):
        print(f"  {name}: {mean[j]}")

print("\nOptimized Parameters:")
for j, name in enumerate(param_names):
    print(f"{name}: {mean[j]}") 