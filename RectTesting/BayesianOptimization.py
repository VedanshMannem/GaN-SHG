import sys, os
import csv

from numpy import delete, real, shape
sys.path.append("C:\\Program Files\\Lumerical\\v251\\api\\python\\") 
sys.path.append(os.path.dirname(__file__)) 

from pprint import pprint
import lumapi # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
from collections import deque
from rect import runSim1
wv = 1.064e-6  # wavelength

iteration_counter = 0

power_history = deque(maxlen=100)

norm = 2.34482413e-13

def objective_function(AlNxSpan, AlNySpan, AlNzSpan, DFTz, theta):

    global iteration_counter
    iteration_counter += 1
    global power_history
    
    try:
        AlNxSpan = float(AlNxSpan) * 1e-6
        AlNySpan = float(AlNySpan) * 1e-6
        AlNzSpan = float(AlNzSpan) * 1e-6
        DFTz = float(DFTz) * 1e-6
        theta = float(theta)

        print(f"Testing parameters: AlNxSpan={AlNxSpan*1e6:.3f}μm, AlNySpan={AlNySpan*1e6:.3f}μm, AlNzSpan={AlNzSpan*1e6:.3f}μm, DFTz={DFTz*1e6:.3f}μm, theta={theta:.1f}° ")

        result = float(runSim1(AlNxSpan, AlNySpan, AlNzSpan, DFTz, theta))

        standardized_power = result / norm

        print(f"Power result: {result} (standardized: {standardized_power})")
    
        log_to_csv(iteration_counter, AlNxSpan, AlNySpan, AlNzSpan, DFTz, theta, result, "success")

        return standardized_power
    
    except Exception as e:
        print(f"Error in simulation: {e}")
        error_msg = str(e)

        log_to_csv(iteration_counter, AlNxSpan, AlNySpan, AlNzSpan, DFTz, theta, 0.0, f"error: {error_msg}")

        return 0.0 

def log_to_csv(iteration, AlNxSpan, AlNySpan, AlNzSpan, DFTz, theta, power, status):
    try:
        with open("second_bayesian_optimization.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iteration, AlNxSpan, AlNySpan, AlNzSpan, DFTz, theta, power, status])
        print(f"Logged iteration {iteration} to CSV")
    except Exception as e:
        print(f"Error logging to CSV: {e}")
        import traceback
        traceback.print_exc()

def load_historical_data(max_trials=50):
    historical_params = []
    historical_objectives = []

    try:
        with open("rectBOp_data.csv", 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            
            count = 0
            for row in reader:
                # Check if row has enough columns and is a successful run
                if len(row) >= 8 and row[8].strip() == "success" and float(row[7]) > 0:
                    try:
                        
                        params = {
                            "AlNxSpan": float(row[4]) * 1e6,  # xspan in micrometers 
                            "AlNySpan": float(row[5]) * 1e6,  # yspan in micrometers
                            "AlNzSpan": float(row[6]) * 1e6,  # zspan in micrometers
                            "DFTz": float(row[3]) * 1e6,     # z in micrometers
                            "theta": 45.0  # Default theta value since it's not in old data
                        }
                        
                        power_value = float(row[7])
                        standardized_power = power_value / norm

                        historical_params.append(params)
                        historical_objectives.append({"power": standardized_power})
                        
                        count += 1
                        if count >= max_trials:
                            break
                            
                    except (ValueError, IndexError) as e:
                        print(f"Skipping row due to parsing error: {e}")
                        continue
        
        print(f"Loaded {len(historical_params)} historical trials for warm-starting")
        return historical_params, historical_objectives
        
    except FileNotFoundError:
        print("No historical data file found (rectBOp_data.csv), starting fresh")
        return [], []
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return [], []

def run_bayesian_optimization():
    
    import os
    if not os.path.exists("second_bayesian_optimization.csv"):
        try:
            with open("second_bayesian_optimization.csv", "a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["iteration", "AlNxSpan_um", "AlNySpan_um", "AlNzSpan_um", "DFTz_um", "theta_deg", "power", "status"])
            print("Initialized second_bayesian_optimization.csv for logging")
        except Exception as e:
            print(f"Error initializing CSV file: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    else:
        print("Using existing second_bayesian_optimization.csv file")
    
    global iteration_counter
    iteration_counter = 0
    
    client = Client()
    
    client.configure_experiment(
        parameters=[
            RangeParameterConfig(
                name="AlNxSpan",
                bounds=(0.5 * wv * 1e6,  2.0 * wv * 1e6),  # Expanded range to match historical data
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="AlNySpan",
                bounds=(0.5 * wv * 1e6,  2.0 * wv * 1e6),  # Expanded range to match historical data
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="AlNzSpan", 
                bounds=(0.5 * wv * 1e6, 2.0 * wv * 1e6),  # Expanded range to match historical data
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="DFTz",
                bounds=(-1.0 * wv * 1e6, 1.0 * wv * 1e6),  # Expanded range to match historical data
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="theta",
                bounds=(30.0, 50.0),  
                parameter_type="float"
            ),
        ],
    )

    client.configure_optimization(objective="power")
    
    historical_params, historical_objectives = load_historical_data(max_trials=60)
    if historical_params and historical_objectives:
        print(f"Warm-starting optimization with {len(historical_params)} historical trials...")
        
        for params, objective in zip(historical_params, historical_objectives):
            trial_index = client.attach_trial(parameters=params)
            client.complete_trial(trial_index=trial_index, raw_data=objective)
            
        print("Historical data injected successfully!")
    else:
        print("No historical data available, starting fresh optimization")
    
    print("Starting Bayesian Optimization...")
    print("=" * 60)
    
    num_iterations = 30  
    
    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        print("-" * 40)
        
        trials = client.get_next_trials(max_trials=1)
        if not trials:
            print("No more trials available")
            break

        for trial_index, parameters in trials.items():
            result = objective_function(
                parameters["AlNxSpan"],
                parameters["AlNySpan"],
                parameters["AlNzSpan"],
                parameters["DFTz"],
                parameters["theta"]
            )
            
            result = float(result)
            client.complete_trial(trial_index = trial_index, raw_data={"power": result})
    
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    
    best_parameters, prediction, index, name = client.get_best_parameterization()

    # Extract the actual prediction value (could be tuple or single value)
    if isinstance(prediction['power'], (tuple, list)):
        best_power = prediction['power'][0]  # Take the first element if it's a tuple/list
    else:
        best_power = prediction['power']

    print(f"\nFinal Best Objective (Power): {best_power:.3f} (standardized) and {(best_power / norm):.6e} (original)")
    print("Final Best Parameters:")

    final_AlNxSpan = float(best_parameters["AlNxSpan"]) * 1e-6
    final_AlNySpan = float(best_parameters["AlNySpan"]) * 1e-6
    final_AlNzSpan = float(best_parameters["AlNzSpan"]) * 1e-6
    final_DFTz = float(best_parameters["DFTz"]) * 1e-6
    final_theta = float(best_parameters["theta"])

    print(f"\nConverted to simulation units:")
    print(f"  AlNxSpan: {final_AlNxSpan:.6e} m")
    print(f"  AlNySpan: {final_AlNySpan:.6e} m")
    print(f"  AlNzSpan: {final_AlNzSpan:.6e} m")
    print(f"  DFTz: {final_DFTz:.6e} m")
    print(f"  theta: {final_theta:.3f}°")

    return best_parameters, prediction['power']

if __name__ == "__main__":
    best_params, best_obj = run_bayesian_optimization()
