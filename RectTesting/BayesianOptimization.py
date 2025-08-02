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
wv = 0.73e-6  # wavelength
num_iterations = 50 

iteration_counter = 0

power_history = deque(maxlen=100)

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

        standardized_power = result 

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
        with open("RectTesting/fixed_BOp.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iteration, AlNxSpan, AlNySpan, AlNzSpan, DFTz, theta, power, status])
        print(f"Logged iteration {iteration} to CSV")
    except Exception as e:
        print(f"Error logging to CSV: {e}")
        import traceback
        traceback.print_exc()

def run_bayesian_optimization():
    
    import os
    if not os.path.exists("fixed_BOp.csv"):
        try:
            with open("fixed_BOp.csv", "a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["iteration", "AlNxSpan_um", "AlNySpan_um", "AlNzSpan_um", "DFTz_um", "theta_deg", "power", "status"])
            print("Initialized fixed_BOp.csv for logging")
        except Exception as e:
            print(f"Error initializing CSV file: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    else:
        print("Using existing fixed_BOp.csv file")

    global iteration_counter
    iteration_counter = 0
    
    client = Client()
    
    client.configure_experiment(
        parameters=[
            RangeParameterConfig(
                name="AlNxSpan",
                bounds=(0.25 * wv * 1e6,  2 * wv * 1e6),  
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="AlNySpan",
                bounds=(0.25 * wv * 1e6,  2 * wv * 1e6), 
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="AlNzSpan", 
                bounds=(0.25 * wv * 1e6, 2 * wv * 1e6),  
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="DFTz",
                bounds=(-wv * 1e6, wv * 1e6),  
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="theta",
                bounds=(20.0, 60.0),  
                parameter_type="float"
            ),
        ],
    )

    client.configure_optimization(objective="power")

    print("Starting Bayesian Optimization...")
    print("=" * 60)
    
    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        print("-" * 40)
        
        trials = client.get_next_trials(max_trials=1)
        if not trials:
            print("No more trials available")
            break

        for trial_index, parameters in trials.items():

            if (abs(parameters["DFTz"]) > 0.5 * parameters["AlNzSpan"]):
                print(f"Skipping trial {trial_index} due to DFTz constraint")
                client.complete_trial(trial_index, raw_data={"power": 0.0})
                continue

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

    if isinstance(prediction['power'], (tuple, list)):
        best_power = prediction['power'][0]  
    else:
        best_power = prediction['power']

    print(f"\nFinal Best Objective (Power): {best_power:.3f} (standardized) and {(best_power):.6e} (original)")
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
