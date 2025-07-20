import sys, os
import csv
from collections import deque
import numpy as np
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from rect import runSim1

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_filename = os.path.join(script_dir, "rectBOp_data.csv")
csv_headers = ["iteration", "xspan", "yspan", "zspan", "theta", "power", "status"]

power_values = []

POWER_MEAN = 9.340652e-14  
POWER_STD = 7.682561e-13

def standardize_power(power_value):
    epsilon = 1e-20
    return (power_value - POWER_MEAN) / max(POWER_STD, epsilon)

def unstandardize_power(standardized_value):
    return standardized_value * POWER_STD + POWER_MEAN

def update_standardization_if_needed(current_powers):
    global POWER_MEAN, POWER_STD
    
    if len(current_powers) >= 20:  # Only update after sufficient data
        current_mean = np.mean(current_powers)
        current_std = np.std(current_powers)
        
        # Check if current data is significantly different (>2x difference)
        mean_ratio = abs(current_mean / POWER_MEAN) if POWER_MEAN > 0 else float('inf')
        std_ratio = abs(current_std / POWER_STD) if POWER_STD > 0 else float('inf')
        
        if mean_ratio > 2 or mean_ratio < 0.5 or std_ratio > 2 or std_ratio < 0.5:
            print("WARNING: Current data significantly different from historical data")
            print(f"Historical - Mean: {POWER_MEAN:.6e}, Std: {POWER_STD:.6e}")
            print(f"Current   - Mean: {current_mean:.6e}, Std: {current_std:.6e}")
            print("Consider updating POWER_MEAN and POWER_STD constants in the code")

with open(csv_filename, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_headers)

iteration_counter = 0

def log_to_csv(iteration, xspan, yspan, zspan, theta, power, status):
    with open(csv_filename, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([iteration, xspan, yspan, zspan, theta, power, status])

def objective_function(xspan, yspan, zspan, theta):
    global iteration_counter, power_values
    iteration_counter += 1
    
    xspan_m = float(xspan) * 1e-6
    yspan_m = float(yspan) * 1e-6
    zspan_m = float(zspan) * 1e-6


    print(f"Testing rect: xspan={xspan:.3f}, yspan={yspan:.3f}, zspan={zspan:.3f}, theta={theta:.3f} (all um)")
    result = runSim1(xspan_m, yspan_m, zspan_m, theta)
    if hasattr(result, 'item'):
        power_value = float(result.item())
    elif isinstance(result, (list, np.ndarray)):
        power_value = float(result[0]) if len(result) > 0 else 0.0
    else:
        power_value = float(result)

    # std the power for better model
    standardized_power = standardize_power(power_value)
    
    # Debug: Print standardization details for first few iterations
    if iteration_counter <= 5:
        print(f"DEBUG - Raw power: {power_value:.6e}")
        print(f"DEBUG - Mean used: {POWER_MEAN:.6e}")
        print(f"DEBUG - Std used: {POWER_STD:.6e}")
        print(f"DEBUG - Standardized: ({power_value:.6e} - {POWER_MEAN:.6e}) / {POWER_STD:.6e} = {standardized_power:.6f}")
    
    power_values.append(power_value)
    
    if len(power_values) >= 10 and len(power_values) % 10 == 0:
        power_array = np.array(power_values)
        mean_power = np.mean(power_array)
        std_power = np.std(power_array)
        min_power = np.min(power_array)
        max_power = np.max(power_array)
        
        # Check if standardization parameters need updating
        update_standardization_if_needed(power_values)
        
        print(f"Power Statistics (last {len(power_values)} trials):")
        print(f"  Mean: {mean_power:.6e}")
        print(f"  Std:  {std_power:.6e}")
        print(f"  Min:  {min_power:.6e}")
        print(f"  Max:  {max_power:.6e}")
        print(f"  Range: {max_power - min_power:.6e}")
        
        standardized_values = [standardize_power(p) for p in power_values[-10:]]
        print(f"Standardized Statistics (last 10 trials):")
        print(f"  Mean: {np.mean(standardized_values):.3f}")
        print(f"  Std:  {np.std(standardized_values):.3f}")
        print(f"  Min:  {np.min(standardized_values):.3f}")
        print(f"  Max:  {np.max(standardized_values):.3f}")

    print(f"Power result: {power_value:.6e} (raw) -> {standardized_power:.3f} (standardized)")
    print(f"RETURNING TO AX: {standardized_power:.6f}")  # Debug what Ax receives
    log_to_csv(iteration_counter, xspan, yspan, zspan, theta, power_value, "success")
    return float(standardized_power) 

def load_historical_data(max_trials=50):
    historical_params = []
    historical_objectives = []
    
    try:
        with open(csv_filename, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)  # Skip header
            
            count = 0
            for row in reader:
                if len(row) >= 9 and row[8] == "success":  # Status is column 8 (0-indexed)
                    try:
                        # Extract only span parameters and add random theta
                        params = {
                            "xspan": float(row[4]),
                            "yspan": float(row[5]),
                            "zspan": float(row[6]),
                            "theta": float(40)  
                        }
                        
                        # Extract and standardize power value (column 7)
                        power_value = float(row[7])
                        standardized_power = standardize_power(power_value)
                        
                        historical_params.append(params)
                        historical_objectives.append({"power": standardized_power})
                        
                        count += 1
                        if count >= max_trials:
                            break
                            
                    except (ValueError, IndexError):
                        continue
        
        print(f"Loaded {len(historical_params)} historical trials for warm-starting")
        return historical_params, historical_objectives
        
    except FileNotFoundError:
        print("No historical data file found, starting fresh")
        return [], []

def run_bayesian_optimization():
    global iteration_counter, power_values
    iteration_counter = 0
    power_values = []  

    client = Client()
    client.configure_experiment(
        parameters=[
            RangeParameterConfig(name="xspan", bounds=(0.5, 3.4), parameter_type="float"),
            RangeParameterConfig(name="yspan", bounds=(0.5, 3.4), parameter_type="float"),
            RangeParameterConfig(name="zspan", bounds=(0.5, 3.4), parameter_type="float"),
            RangeParameterConfig(name="theta", bounds=(0, 90), parameter_type="float"),
        ],
    )

    num_iterations = 60

    # set to "-power" to minimize if ever needed
    client.configure_optimization(
        objective="power"
    )
    
    # Load and inject historical data for warm-starting
    historical_params, historical_objectives = load_historical_data(max_trials=100)
    if historical_params and historical_objectives:
        print(f"Warm-starting optimization with {len(historical_params)} historical trials...")
        
        # Add historical trials to the client
        for params, objective in zip(historical_params, historical_objectives):
            trial_index = client.attach_trial(parameters=params)
            client.complete_trial(trial_index=trial_index, raw_data=objective)
            
        print("Historical data injected successfully!")
    else:
        print("No historical data available, starting fresh optimization")
    
    print("Starting Rect Bayesian Optimization with Standardized Power Values...")
    print(f"Historical data - Mean: {POWER_MEAN:.6e}, Std: {POWER_STD:.6e}")
    print("Ax will receive standardized values (mean=0, std=1)")
    print("=" * 60)
    
    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        print("-" * 40)
        trials = client.get_next_trials(max_trials=1)
        if not trials:
            print("No more trials available")
            break
        for trial_index, parameters in trials.items():
            result = objective_function(
                parameters["xspan"], parameters["yspan"], parameters["zspan"], parameters["theta"]
            )
            client.complete_trial(trial_index=trial_index, raw_data={"power": float(result)})

    print("\n" + "=" * 60)
    print("Optimization Complete!")

    # Final best
    try:
        best_parameters, prediction, index, name = client.get_best_parameterization() 
        
        if isinstance(prediction["power"], (list, tuple, np.ndarray)):
            best_objective = float(prediction["power"][0])  
        else:
            best_objective = float(prediction["power"])  
        
        best_power_original = unstandardize_power(best_objective)
        print(f"\nFinal Best Objective (Power): {best_objective:.3f} (standardized) = {best_power_original:.6e} (original)")
        print(f"Final Best Rect: xspan={best_parameters['xspan']:.3f}um, yspan={best_parameters['yspan']:.3f}um, zspan={best_parameters['zspan']:.3f}um, theta={best_parameters['theta']:.1f}°")
        print(f"Converted to meters: xspan={best_parameters['xspan']*1e-6:.6e}, yspan={best_parameters['yspan']*1e-6:.6e}, zspan={best_parameters['zspan']*1e-6:.6e}")
        return best_parameters, best_power_original  

    except Exception as e:
        print(f"Error getting final best parameters: {e}")
        return None, None

if __name__ == "__main__":
    run_bayesian_optimization() 