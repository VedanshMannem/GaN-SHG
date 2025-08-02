# Test script for best Bayesian optimization parameters
import sys, os
sys.path.append(os.path.dirname(__file__))

# Import your simulation function here
# from triangle import runSim1  # Adjust import as needed

# Best parameters found from Bayesian optimization
best_x1 = -7.300000e-13  # meters
best_x2 = 2.632116e-13  # meters  
best_x3 = -1.350506e-13  # meters
best_y1 = 7.300000e-13  # meters
best_y2 = -1.046706e-13  # meters
best_y3 = -4.300044e-13  # meters
best_AlNzSpan = 1.460000e-12  # meters
best_DFTz = -2.235726e-15  # meters
best_theta = 30.000  # degrees

print("Testing best parameters from Bayesian optimization:")
print(f"Expected power: 1.406759")
print(f"Iteration: 40")

# Uncomment to test:
# result = runSim1(best_x1, best_x2, best_x3, best_y1, best_y2, best_y3, best_AlNzSpan, best_DFTz, best_theta)
# print(f"Actual power: {result}")
