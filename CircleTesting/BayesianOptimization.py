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
from circle import runSim1
wv = 1.064e-6  # wavelength

iteration_counter = 0

power_history = deque(maxlen=100)

norm = 2.34482413e-13

def objective_function(radius, AlNzSpan, DFTz, theta):

    global iteration_counter
    iteration_counter += 1
    global power_history
    
    try:
        radius = float(radius) * 1e-6
        AlNzSpan = float(AlNzSpan) * 1e-6
        DFTz = float(DFTz) * 1e-6
        theta = float(theta)

        print(f"Testing parameters: radius={radius*1e6:.3f}μm, AlNzSpan={AlNzSpan*1e6:.3f}μm, DFTz={DFTz*1e6:.3f}μm, theta={theta:.1f}° ")
        
        constraint_satisfied = abs(DFTz) <= 0.5 * AlNzSpan
        print(f"  Constraint check: abs({DFTz:.3e}) <= 0.5*{AlNzSpan:.3e} = {constraint_satisfied}")

        result = float(runSim1(radius, AlNzSpan, DFTz, theta))

        standardized_power = result / norm

        print(f"Power result: {result} (standardized: {standardized_power})")
        
        log_to_csv(iteration_counter, radius, AlNzSpan, DFTz, theta, result, "success")

        return standardized_power
    
    except Exception as e:
        print(f"Error in simulation: {e}")
        error_msg = str(e)

        log_to_csv(iteration_counter, radius, AlNzSpan, DFTz, theta, 0.0, f"error: {error_msg}")

        return 0.0 

def log_to_csv(iteration, radius, AlNzSpan, DFTz, theta, power, status):
    try:
        with open("second_bayesian_optimization.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iteration, radius, AlNzSpan, DFTz, theta, power, status])
        print(f"Logged iteration {iteration} to CSV")
    except Exception as e:
        print(f"Error logging to CSV: {e}")
        import traceback
        traceback.print_exc()

# def load_historical_data(max_trials=50):
#     historical_params = []
#     historical_objectives = []
    
#     try:
#         with open("baycircle_data.csv", 'r') as file:
#             reader = csv.reader(file)
#             headers = next(reader)
            
#             count = 0
#             for row in reader:
#                 if len(row) >= 6 and row[5] != "0.0" and "success" in row[-1]:
#                     try:
#                         params = {
#                             "radius": float(row[1]) * 1e6,  
#                             "AlNzSpan": float(row[2]) * 1e6, 
#                             "theta": float(row[4])
#                         }
                        
#                         power_value = float(row[5])
#                         standardized_power = power_value / norm
#
#                         historical_params.append(params)
#                         historical_objectives.append({"power": standardized_power})
                        
#                         count += 1
#                         if count >= max_trials:
#                             break
                            
#                     except (ValueError, IndexError):
#                         continue
        
#         print(f"Loaded {len(historical_params)} historical trials for warm-starting")
#         return historical_params, historical_objectives
        
#     except FileNotFoundError:
#         print("No historical data file found, starting fresh")
#         return [], []

def run_bayesian_optimization():
    try:
        with open("second_bayesian_optimization.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["iteration", "radius_um", "AlNzSpan_um", "DFTz_um", "theta_deg", "power", "status"])
        print("Initialized second_bayesian_optimization.csv for logging")
    except Exception as e:
        print(f"Error initializing CSV file: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    global iteration_counter
    iteration_counter = 0
    
    client = Client()
    
    client.configure_experiment(
        parameters=[
            RangeParameterConfig(
                name="radius",
                bounds=(0.25 * wv * 1e6,  2 * wv * 1e6),  
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="AlNzSpan", 
                bounds=(0.5 * wv * 1e6, 3 * wv * 1e6),  
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="DFTz",
                bounds=(-1.5 * wv * 1e6, 1.5 * wv * 1e6),  
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
    
    # historical_params, historical_objectives = load_historical_data(max_trials=60)
    # if historical_params and historical_objectives:
    #     print(f"Warm-starting optimization with {len(historical_params)} historical trials...")
        
    #     for params, objective in zip(historical_params, historical_objectives):
    #         trial_index = client.attach_trial(parameters=params)
    #         client.complete_trial(trial_index=trial_index, raw_data=objective)
            
    #     print("Historical data injected successfully!")
    # else:
    #     print("No historical data available, starting fresh optimization")
    
    print("Starting Bayesian Optimization...")
    print("=" * 60)
    
    num_iterations = 30  
    
    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        print("-" * 40)
        
        trial_successful = False
        attempts = 0
        max_attempts = 10  

        while not trial_successful and attempts < max_attempts:
            attempts += 1
            
            trials = client.get_next_trials(max_trials=1)
            if not trials:
                print("No more trials available")
                break
                
            for trial_index, parameters in trials.items():
                if abs(parameters["DFTz"]) > 0.5 * parameters["AlNzSpan"]:
                    print(f"Attempt {attempts}: Skipping trial {trial_index} due to DFTz constraint")
                    print(f"  DFTz={parameters['DFTz']:.3e}, AlNzSpan={parameters['AlNzSpan']:.3e}")
                    print(f"  Need: abs(DFTz) <= 0.5*AlNzSpan = {0.5 * parameters['AlNzSpan']:.3e}")
                    try:
                        client.abandon_trial(trial_index=trial_index)
                    except:
                        client.complete_trial(trial_index=trial_index, raw_data={"power": -1e-10})
                    continue
                else:
                    print(f"Valid trial found on attempt {attempts}")
                    print(f"  DFTz={parameters['DFTz']:.3e}, AlNzSpan={parameters['AlNzSpan']:.3e}")
                    print(f"  Constraint satisfied: abs(DFTz) <= 0.5*AlNzSpan")
                    
                    result = objective_function(
                        parameters["radius"],
                        parameters["AlNzSpan"],
                        parameters["DFTz"],
                        parameters["theta"]
                    )
                    
                    result = float(result)
                    client.complete_trial(trial_index = trial_index, raw_data={"power": result})
                    trial_successful = True
                    break
        
        if not trial_successful:
            print(f"Warning: Could not find valid parameters after {max_attempts} attempts for iteration {i+1}")
            break
    
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    
    best_parameters, prediction, index, name = client.get_best_parameterization()

    print(f"\nFinal Best Objective (Power): {prediction['power']:.3f} (standardized) and {(prediction['power'] / norm):.6e} (original)")
    print("Final Best Parameters:")

    final_radius = float(best_parameters["radius"]) * 1e-6
    final_AlNzSpan = float(best_parameters["AlNzSpan"]) * 1e-6
    final_DFTz = float(best_parameters["DFTz"]) * 1e-6
    final_theta = float(best_parameters["theta"])

    print(f"\nConverted to simulation units:")
    print(f"  radius: {final_radius:.6e} m")
    print(f"  AlNzSpan: {final_AlNzSpan:.6e} m") 
    print(f"  DFTz: {final_DFTz:.6e} m")
    print(f"  theta: {final_theta:.3f}°")

    return best_parameters, prediction['power']

if __name__ == "__main__":
    best_params, best_obj = run_bayesian_optimization()
