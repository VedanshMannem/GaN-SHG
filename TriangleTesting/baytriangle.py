import sys, os
import csv

from numpy import delete, real, shape, sqrt
sys.path.append("C:\\Program Files\\Lumerical\\v251\\api\\python\\") 
sys.path.append(os.path.dirname(__file__)) 

from pprint import pprint
import lumapi # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from collections import deque

chi1 = 5.3458
theta = 40 # angle of plane wave
c = 299792458

def runSim1(x1, x2, x3, y1, y2, y3):

    V = np.array([
        [x1, y1],
        [x2, y2],
        [x3, y3]
    ])

    mesh = 0.085e-6

    GaNx = 0
    GaNy = 0
    GaNz = 0
    GaNzSpan = 1.064e-6

    Sapx = 0
    Sapy = 0
    SapSpan = 4e-6 # x, y span
    SapzSpan = 6e-6 # z span
    Sapz = -0.5 * (GaNzSpan + SapzSpan)

    PlaneZ = GaNz - 0.5 * GaNzSpan - 1.064e-6
    PlaneX = 0
    PlaneY = 0

    FDTDx = 0
    FDTDy = 0
    FDTDspan = 4e-6 # x & y span
    FDTDzMin = PlaneZ - 1.064e-6
    FDTDzMax = GaNz + GaNzSpan * 0.5 + 1.064e-6 # 0.55 microns for the wavelength

    GaNDFTx = 0
    GaNDFTy = 0
    GaNDFTz = 0 

    GaNDFTx2 = 0
    GaNDFTy2 = 0
    GaNDFTz2 = GaNz + 0.5 * GaNzSpan + 1.064e-6

    FDTDzMin2 = GaNz - 0.5 * GaNzSpan - 1.064e-6
    FDTDzMax2 = GaNDFTz2 + 1.064e-6

    fdtd = lumapi.FDTD(hide = True)

    fdtd.eval("q=[2.1297;2.1297;2.1712];setmaterial(addmaterial(\"(n,k) Material\"), \"name\", \"GaN\");setmaterial(\"GaN\", \"Anisotropy\", 1);setmaterial(\"GaN\", \"Refractive Index\", q);") 

    fdtd.addfdtd()
    fdtd.addmesh()

    fdtd.addplane()
    fdtd.set("name", "PlaneWave")

    fdtd.addtriangle()
    fdtd.set("name", "GaNfilm")
    fdtd.set("material", "GaN")

    fdtd.addrect()
    fdtd.set("name", "Sapphire")
    fdtd.set("material", "Al2O3 - Palik")

    fdtd.adddftmonitor()
    fdtd.set("name", "GaNDFT") # GaNDFT in Lumerical

    # material = GaN
    # rect structure = GaNfilm

    configuration = (
        ("PlaneWave", (
                    ("x", PlaneX),
                    ("y", PlaneY),
                    ("z", PlaneZ),
                    ("x span", SapSpan),
                    ("y span", SapSpan),
                    ("angle theta", theta),
                    ("wavelength start", 1.064e-6),
                    ("wavelength stop", 1.064e-6))),

        ("GaNfilm", (
                    ("x",GaNx),
                    ("y",GaNy),
                    ("z",GaNz),
                    ("z span", GaNzSpan),
                    ("vertices", V))),

        ("mesh", (("dx", mesh),
                  ("dy", mesh),
                  ("dz", mesh),
                  ("based on a structure", True),
                  ("structure", "GaNfilm"))),

        ("FDTD", (("x",FDTDx),
                  ("y",FDTDy),
                  ("x span", FDTDspan),
                  ("y span", FDTDspan),
                  ("z min", FDTDzMin),
                  ("z max", FDTDzMax),
                  ("x min bc", "periodic"),
                  ("y min bc", "periodic"))),

        ("Sapphire", ( 
                  ("x",Sapx),
                  ("y",Sapy),
                  ("z",Sapz),
                  ("x span", SapSpan),
                  ("y span", SapSpan),
                  ("z span", SapzSpan),
                  ("material", "Al2O3 - Palik"))),
                  
        ("GaNDFT", (
                    ("x", GaNDFTx),
                    ("y", GaNDFTy),
                    ("z", GaNDFTz),
                    ("x span", FDTDspan),
                    ("y span", FDTDspan)))
    )

    for obj, parameters in configuration:
       for k, v in parameters:
           fdtd.setnamed(obj, k, v)

    fdtd.save("lumapitesting")
    fdtd.run()

    # Full eval script to get the imported source
    fdtd.eval("E2 = rectilineardataset(\"EM Fields\", getresult(\"GaNDFT\", \"x\"), getresult(\"GaNDFT\", \"y\"), getresult(\"GaNDFT\", \"z\"));")
    fdtd.eval("chi1 = 5.3458;")
    fdtd.eval("Ex= getresult(\"GaNDFT\", \"Ex\");Ey= getresult(\"GaNDFT\", \"Ey\");Ez= getresult(\"GaNDFT\", \"Ez\");")
    fdtd.eval("E2x = (2 * 11.33 * Ez * Ex) / chi1; E2y = (2 * 11.33 * Ez * Ey) / chi1; E2z = (11.33 * (Ex ^ 2 + Ey ^ 2) - 22.66 * Ez ^ 2) / chi1;")
    fdtd.eval("E2.addparameter(\"lambda\", 299792458/getresult(\"GaNDFT\", \"f\"), \"f\", getresult(\"GaNDFT\", \"f\"));")
    fdtd.eval("E2.addattribute(\"E\", E2x, E2y, E2z);")

    fdtd.switchtolayout()
    fdtd.select("PlaneWave")
    fdtd.delete()

    fdtd.select("FDTD")
    fdtd.delete()

    fdtd.eval(f"addimportedsource; importdataset(E2);set(\"name\", \"source2\");set(\"x\", {GaNDFTx});set(\"y\", {GaNDFTy});set(\"z\", {GaNDFTz});set(\"injection axis\", \"z\");set(\"direction\", \"forward\");")

    fdtd.addfdtd()

    fdtd.adddftmonitor()
    fdtd.set("name", "GaNDFT2")

    configuration2 = (
        ("FDTD", (("x",FDTDx),
                  ("y",FDTDy),
                  ("x span", FDTDspan),
                  ("y span", FDTDspan),
                  ("z min", FDTDzMin2),
                  ("z max", FDTDzMax2),
                  ("x min bc", "periodic"),
                  ("y min bc", "periodic"))),
        ("GaNDFT2", (
                    ("x", GaNDFTx2),
                    ("y", GaNDFTy2),
                    ("z", GaNDFTz2),
                    ("x span", FDTDspan),
                    ("y span", FDTDspan))),
    )

    for obj, parameters in configuration2:
       for k, v in parameters:
           fdtd.setnamed(obj, k, v)
    
    fdtd.save("triangle-test2")
    fdtd.run()

    result = fdtd.getresult("GaNDFT2", "power")
    return real(result)


def calculate_triangle_heights(x1, x2, x3, y1, y2, y3):
    
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    
    side1 = sqrt((x2 - x1)**2 + (y2 - y1)**2)  
    side2 = sqrt((x3 - x2)**2 + (y3 - y2)**2)  
    side3 = sqrt((x1 - x3)**2 + (y1 - y3)**2)  
    
    height1 = 2 * area / side1 if side1 > 0 else 0 
    height2 = 2 * area / side2 if side2 > 0 else 0 
    height3 = 2 * area / side3 if side3 > 0 else 0 
    
    return min(height1, height2, height3)

def check_triangle_constraints(x1, x2, x3, y1, y2, y3, min_height_um=0.532):
    min_height = calculate_triangle_heights(x1, x2, x3, y1, y2, y3)
    return min_height >= min_height_um * 1e-6  

mean = 1.450726e-13
std = 9.444022e-13

def standardize_power(power_value):
    power_value - mean
    standardized_power = (power_value - mean) / std
    return standardized_power

def unstandardize_power(standardized_power):
    return standardized_power * std + mean

iteration_counter = 0
power_history = deque(maxlen=100)

def objective_function(x1, x2, x3, y1, y2, y3):
    global iteration_counter
    iteration_counter += 1
    global power_history
    
    try:
        x1_m = float(x1) * 1e-6
        x2_m = float(x2) * 1e-6  
        x3_m = float(x3) * 1e-6
        y1_m = float(y1) * 1e-6
        y2_m = float(y2) * 1e-6
        y3_m = float(y3) * 1e-6
        
        print(f"Testing triangle: ({x1:.3f}, {y1:.3f}), ({x2:.3f}, {y2:.3f}), ({x3:.3f}, {y3:.3f}) μm")
        
        result = runSim1(x1_m, x2_m, x3_m, y1_m, y2_m, y3_m)
        
        if hasattr(result, 'item'):
            power_value = float(result.item())
        elif isinstance(result, (list, np.ndarray)):
            power_value = float(result[0]) if len(result) > 0 else 0.0
        else:
            power_value = float(result)

        standardized_power = standardize_power(power_value)

        print(f"Power result: {power_value} (standardized: {standardized_power})")
        
        log_to_csv(iteration_counter, x1, x2, x3, y1, y2, y3, power_value, "success")
        
        return standardized_power
        
    except Exception as e:
        print(f"Error in simulation: {e}")
        error_msg = str(e)
        
        log_to_csv(iteration_counter, x1, x2, x3, y1, y2, y3, 0.0, f"error: {error_msg}")
        
        return 0.0

def log_to_csv(iteration, x1, x2, x3, y1, y2, y3, power, status):
    try:
        with open("baytriangle_data.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iteration, x1, x2, x3, y1, y2, y3, power, status])
    except Exception as e:
        print(f"Error logging to CSV: {e}")

def load_historical_data(max_trials=50):
    """Load historical data from triRL_data.csv and return parameters and objectives"""
    historical_params = []
    historical_objectives = []
    
    try:
        with open("triRL_data.csv", 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)  # Skip header
            
            count = 0
            for row in reader:
                if len(row) >= 9 and row[8] != "0.0" and "success" in str(row[-1]):  # Skip errors and zero power
                    try:
                        # Extract triangle vertex parameters (convert from meters to micrometers)
                        params = {
                            "x1": float(row[2]) * 1e6,  # Convert m to μm
                            "x2": float(row[3]) * 1e6,  # Convert m to μm  
                            "x3": float(row[4]) * 1e6,  # Convert m to μm
                            "y1": float(row[5]) * 1e6,  # Convert m to μm
                            "y2": float(row[6]) * 1e6,  # Convert m to μm
                            "y3": float(row[7]) * 1e6   # Convert m to μm
                        }
                        
                        # Extract power value (column 8: power_output)
                        power_value = float(row[8])
                        
                        historical_params.append(params)
                        historical_objectives.append({"power": power_value})
                        
                        count += 1
                        if count >= max_trials:
                            break
                            
                    except (ValueError, IndexError):
                        continue
        
        print(f"Loaded {len(historical_params)} historical trials for warm-starting")
        return historical_params, historical_objectives
        
    except FileNotFoundError:
        print("No triRL_data.csv file found, starting fresh")
        return [], []

def run_bayesian_optimization():
    try:
        with open("baytriangle_data.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["iteration", "x1_um", "x2_um", "x3_um", "y1_um", "y2_um", "y3_um", "power", "status"])
        print("Initialized baytriangle_data.csv for logging")
    except Exception as e:
        print(f"Error initializing CSV file: {e}")
    
    global iteration_counter
    iteration_counter = 0
    
    client = Client()
    
    client.configure_experiment(
        parameters=[
            RangeParameterConfig(
                name="x1",
                bounds=(-1.6, 1.6),  
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="x2",
                bounds=(-1.6, 1.6),  
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="x3",
                bounds=(-1.6, 1.6),  
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="y1",
                bounds=(-1.6, 1.6),  
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="y2",
                bounds=(-1.6, 1.6),  
                parameter_type="float"
            ),
            RangeParameterConfig(
                name="y3",
                bounds=(-1.6, 1.6),
                parameter_type="float"
            ),
        ],
    )

    client.configure_optimization(objective="power")
    
    # Load and inject historical data for warm-starting
    historical_params, historical_objectives = load_historical_data(max_trials=30)
    if historical_params and historical_objectives:
        print(f"Warm-starting optimization with {len(historical_params)} historical trials...")
        
        # Add historical trials to the client
        for params, objective in zip(historical_params, historical_objectives):
            trial_index = client.attach_trial(parameters=params)
            client.complete_trial(trial_index=trial_index, raw_data=objective)
            
        print("Historical data injected successfully!")
    else:
        print("No historical data available, starting fresh optimization")
    
    print("Starting Triangle Bayesian Optimization...")
    print("=" * 60)
    
    num_iterations = 50  
    
    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        print("-" * 40)
        
        trials = client.get_next_trials(max_trials=1)
        if not trials:
            print("No more trials available")
            break

        for trial_index, parameters in trials.items():
            result = objective_function(
                parameters["x1"],
                parameters["x2"], 
                parameters["x3"],
                parameters["y1"],
                parameters["y2"],
                parameters["y3"]
            )
            
            client.complete_trial(trial_index=trial_index, raw_data={"power": float(result)})
        
        
    
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    
    # final best parameters
    try:
            best_parameters, prediction, index, name = client.get_best_parameterization() 
            
            if isinstance(prediction["power"], (list, tuple, np.ndarray)):
                best_objective = float(prediction["power"][0])  
            else:
                best_objective = float(prediction["power"])  

            best_objective = best_parameters[0]["power"] 
            print(f"Current best objective: {best_objective}")
            print(f"Current best triangle coordinates (μm):")
            print(f"  Vertex 1: ({best_parameters[0]['x1']:.3f}, {best_parameters[0]['y1']:.3f})")
            print(f"  Vertex 2: ({best_parameters[0]['x2']:.3f}, {best_parameters[0]['y2']:.3f})")
            print(f"  Vertex 3: ({best_parameters[0]['x3']:.3f}, {best_parameters[0]['y3']:.3f})")
            
            # Calculate and display minimum height
            min_height = calculate_triangle_heights(
                best_parameters[0]['x1']*1e-6, best_parameters[0]['x2']*1e-6, best_parameters[0]['x3']*1e-6,
                best_parameters[0]['y1']*1e-6, best_parameters[0]['y2']*1e-6, best_parameters[0]['y3']*1e-6
            )
            print(f"  Minimum height: {min_height*1e6:.3f} μm")

    except Exception as e:
        print(f"Could not get best parameters yet: {e}")

if __name__ == "__main__":
    best_params, best_obj = run_bayesian_optimization()
