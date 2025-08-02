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
from triangle import runSim1
wv = 0.73e-6  # wavelength

iteration_counter = 0

power_history = deque(maxlen=100)

def objective_function(x1, x2, x3, y1, y2, y3, AlNzSpan, DFTz, theta):

    global iteration_counter
    iteration_counter += 1
    global power_history
    
    try:
        x1 = float(x1) * 1e-6
        x2 = float(x2) * 1e-6
        x3 = float(x3) * 1e-6
        y1 = float(y1) * 1e-6
        y2 = float(y2) * 1e-6
        y3 = float(y3) * 1e-6
        AlNzSpan = float(AlNzSpan) * 1e-6
        DFTz = float(DFTz) * 1e-6
        theta = float(theta)

        print(f"Testing parameters: x1={x1*1e6:.3f}μm, x2={x2*1e6:.3f}μm, x3={x3*1e6:.3f}μm, y1={y1*1e6:.3f}μm, y2={y2*1e6:.3f}μm, y3={y3*1e6:.3f}μm, AlNzSpan={AlNzSpan*1e6:.3f}μm, DFTz={DFTz*1e6:.3f}μm, theta={theta:.1f}° ")
        
        # Debug: Check constraint again
        constraint_satisfied = abs(DFTz) <= 0.5 * AlNzSpan
        print(f"  Constraint check: abs({DFTz:.3e}) <= 0.5*{AlNzSpan:.3e} = {constraint_satisfied}")

        result = float(runSim1(x1, x2, x3, y1, y2, y3, AlNzSpan, DFTz, theta))

        standardized_power = result

        print(f"Power result: {result} (standardized: {standardized_power})")
        
        log_to_csv(iteration_counter, x1, x2, x3, y1, y2, y3, AlNzSpan, DFTz, theta, result, "success")

        return standardized_power
    
    except Exception as e:
        print(f"Error in simulation: {e}")
        error_msg = str(e)
        log_to_csv(iteration_counter, x1, x2, x3, y1, y2, y3, AlNzSpan, DFTz, theta, 0.0, f"error: {error_msg}")
        return 0.0 

def log_to_csv(iteration, x1, x2, x3, y1, y2, y3, AlNzSpan, DFTz, theta, power, status):
    try:
        with open("fixed_BOp.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iteration, x1, x2, x3, y1, y2, y3, AlNzSpan, DFTz, theta, power, status])
        print(f"Logged iteration {iteration} to CSV")
    except Exception as e:
        print(f"Error logging to CSV: {e}")
        import traceback
        traceback.print_exc()

def run_bayesian_optimization():
    try:
        with open("fixed_BOp.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["iteration", "x1_um", "x2_um", "x3_um", "y1_um", "y2_um", "y3_um", "AlNzSpan_um", "DFTz_um", "theta_deg", "power", "status"])
        print("Initialized fixed_BOp.csv for logging")
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
                name="x1",
                bounds=(-wv * 1e6,  wv * 1e6),
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="x2",
                bounds=(-wv * 1e6, wv * 1e6),
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="x3",
                bounds=(-wv * 1e6, wv * 1e6),
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="y1",
                bounds=(-wv * 1e6,  wv * 1e6),
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="y2",
                bounds=(-wv * 1e6, wv * 1e6),
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="y3",
                bounds=(-wv * 1e6, wv * 1e6),
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="AlNzSpan",
                bounds=(0.5 * wv * 1e6, 2 * wv * 1e6),
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="DFTz",
                bounds=(-1 * wv * 1e6, wv * 1e6),  
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

    print("Starting Bayesian Optimization...")
    print("=" * 60)
    
    num_iterations = 50
     
    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        print("-" * 40) 
        
        # Keep trying until we get a valid trial
        trial_successful = False
        attempts = 0
        max_attempts = 10  # Prevent infinite loops
        
        while not trial_successful and attempts < max_attempts:
            attempts += 1
            
            trials = client.get_next_trials(max_trials=1)
            if not trials:
                print("No more trials available")
                break
                
            for trial_index, parameters in trials.items():
                # Check if the constraint is satisfied: abs(DFTz) <= 0.5 * AlNzSpan
                if abs(parameters["DFTz"]) > 0.5 * parameters["AlNzSpan"]:
                    print(f"Attempt {attempts}: Skipping trial {trial_index} due to DFTz constraint")
                    print(f"  DFTz={parameters['DFTz']:.3e}, AlNzSpan={parameters['AlNzSpan']:.3e}")
                    print(f"  Need: abs(DFTz) <= 0.5*AlNzSpan = {0.5 * parameters['AlNzSpan']:.3e}")
                    # Abandon this trial instead of completing it
                    try:
                        client.abandon_trial(trial_index=trial_index)
                    except:
                        # If abandon doesn't work, complete with very small negative value
                        client.complete_trial(trial_index=trial_index, raw_data={"power": -1e-10})
                    continue
                else:
                    print(f"Valid trial found on attempt {attempts}")
                    print(f"  DFTz={parameters['DFTz']:.3e}, AlNzSpan={parameters['AlNzSpan']:.3e}")
                    print(f"  Constraint satisfied: abs(DFTz) <= 0.5*AlNzSpan")
                    
                    # Valid trial - run the simulation
                    result = objective_function(
                        parameters["x1"],
                        parameters["x2"],
                        parameters["x3"],
                        parameters["y1"],
                        parameters["y2"],
                        parameters["y3"],
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

    if isinstance(prediction['power'], (tuple, list)):
        best_power = prediction['power'][0] 
    else:
        best_power = prediction['power']

    print(f"\nFinal Best Objective (Power): {best_power:.3f} (standardized) and {(best_power):.6e} (original)")
    print("Final Best Parameters:")

    final_x1 = float(best_parameters["x1"]) * 1e-6
    final_x2 = float(best_parameters["x2"]) * 1e-6
    final_x3 = float(best_parameters["x3"]) * 1e-6
    final_y1 = float(best_parameters["y1"]) * 1e-6
    final_y2 = float(best_parameters["y2"]) * 1e-6
    final_y3 = float(best_parameters["y3"]) * 1e-6
    final_AlNzSpan = float(best_parameters["AlNzSpan"]) * 1e-6
    final_DFTz = float(best_parameters["DFTz"]) * 1e-6
    final_theta = float(best_parameters["theta"])

    print(f"\nConverted to simulation units:")
    print(f"  x1: {final_x1:.6e} m")
    print(f"  x2: {final_x2:.6e} m")
    print(f"  x3: {final_x3:.6e} m")
    print(f"  y1: {final_y1:.6e} m")
    print(f"  y2: {final_y2:.6e} m")
    print(f"  y3: {final_y3:.6e} m")
    print(f"  AlNzSpan: {final_AlNzSpan:.6e} m")
    print(f"  DFTz: {final_DFTz:.6e} m")
    print(f"  theta: {final_theta:.3f}°")

    return best_parameters, prediction['power']

if __name__ == "__main__":
    best_params, best_obj = run_bayesian_optimization()
