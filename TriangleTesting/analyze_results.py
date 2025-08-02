import pandas as pd
import numpy as np

def analyze_bayesian_results(csv_file):
    """
    Analyze Bayesian optimization results from CSV file to find best parameters
    even if the optimization was interrupted or encountered errors.
    """
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} iterations from {csv_file}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # Filter only successful runs
    successful_runs = df[df['status'] == 'success']
    print(f"Found {len(successful_runs)} successful runs out of {len(df)} total")
    
    if len(successful_runs) == 0:
        print("No successful runs found!")
        return None
    
    # Find the best result (highest power)
    best_idx = successful_runs['power'].idxmax()
    best_result = successful_runs.loc[best_idx]
    
    print("\n" + "="*60)
    print("BEST RESULT FROM BAYESIAN OPTIMIZATION")
    print("="*60)
    print(f"Iteration: {best_result['iteration']}")
    print(f"Power: {best_result['power']:.6f}")
    print(f"Status: {best_result['status']}")
    
    print(f"\nBest Parameters:")
    print(f"  x1: {best_result['x1_um']:.6e} μm")
    print(f"  x2: {best_result['x2_um']:.6e} μm") 
    print(f"  x3: {best_result['x3_um']:.6e} μm")
    print(f"  y1: {best_result['y1_um']:.6e} μm")
    print(f"  y2: {best_result['y2_um']:.6e} μm")
    print(f"  y3: {best_result['y3_um']:.6e} μm")
    print(f"  AlNzSpan: {best_result['AlNzSpan_um']:.6e} μm")
    print(f"  DFTz: {best_result['DFTz_um']:.6e} μm")
    print(f"  theta: {best_result['theta_deg']:.3f}°")
    
    # Convert to simulation units (meters)
    print(f"\nConverted to simulation units (meters):")
    print(f"  x1: {best_result['x1_um'] * 1e-6:.6e} m")
    print(f"  x2: {best_result['x2_um'] * 1e-6:.6e} m")
    print(f"  x3: {best_result['x3_um'] * 1e-6:.6e} m")
    print(f"  y1: {best_result['y1_um'] * 1e-6:.6e} m")
    print(f"  y2: {best_result['y2_um'] * 1e-6:.6e} m")
    print(f"  y3: {best_result['y3_um'] * 1e-6:.6e} m")
    print(f"  AlNzSpan: {best_result['AlNzSpan_um'] * 1e-6:.6e} m")
    print(f"  DFTz: {best_result['DFTz_um'] * 1e-6:.6e} m")
    print(f"  theta: {best_result['theta_deg']:.3f}°")
    
    # Show top 5 results
    top_5 = successful_runs.nlargest(5, 'power')
    print(f"\n" + "="*60)
    print("TOP 5 RESULTS")
    print("="*60)
    for i, (idx, row) in enumerate(top_5.iterrows(), 1):
        print(f"{i}. Iteration {row['iteration']}: Power = {row['power']:.6f}")
    
    # Show some statistics
    print(f"\n" + "="*60)
    print("OPTIMIZATION STATISTICS")
    print("="*60)
    print(f"Total iterations: {len(df)}")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Success rate: {len(successful_runs)/len(df)*100:.1f}%")
    print(f"Best power: {successful_runs['power'].max():.6f}")
    print(f"Worst power: {successful_runs['power'].min():.6f}")
    print(f"Average power: {successful_runs['power'].mean():.6f}")
    print(f"Power improvement: {(successful_runs['power'].max() - successful_runs['power'].min()) / successful_runs['power'].min() * 100:.1f}%")
    
    return best_result

def create_test_script(best_params, script_name="test_best_params.py"):
    """
    Create a test script to verify the best parameters found
    """
    script_content = f'''# Test script for best Bayesian optimization parameters
import sys, os
sys.path.append(os.path.dirname(__file__))

# Import your simulation function here
# from triangle import runSim1  # Adjust import as needed

# Best parameters found from Bayesian optimization
best_x1 = {best_params['x1_um'] * 1e-6:.6e}  # meters
best_x2 = {best_params['x2_um'] * 1e-6:.6e}  # meters  
best_x3 = {best_params['x3_um'] * 1e-6:.6e}  # meters
best_y1 = {best_params['y1_um'] * 1e-6:.6e}  # meters
best_y2 = {best_params['y2_um'] * 1e-6:.6e}  # meters
best_y3 = {best_params['y3_um'] * 1e-6:.6e}  # meters
best_AlNzSpan = {best_params['AlNzSpan_um'] * 1e-6:.6e}  # meters
best_DFTz = {best_params['DFTz_um'] * 1e-6:.6e}  # meters
best_theta = {best_params['theta_deg']:.3f}  # degrees

print("Testing best parameters from Bayesian optimization:")
print(f"Expected power: {best_params['power']:.6f}")
print(f"Iteration: {best_params['iteration']}")

# Uncomment to test:
# result = runSim1(best_x1, best_x2, best_x3, best_y1, best_y2, best_y3, best_AlNzSpan, best_DFTz, best_theta)
# print(f"Actual power: {{result}}")
'''
    
    with open(script_name, 'w') as f:
        f.write(script_content)
    print(f"\nTest script created: {script_name}")

def fit_surrogate_model(df):
    """
    Fit a simple surrogate model to predict optimal parameters beyond sampled points.
    This mimics what Bayesian optimization would do internally.
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        from sklearn.preprocessing import StandardScaler
        import warnings
        warnings.filterwarnings('ignore')
        
        successful_runs = df[df['status'] == 'success']
        if len(successful_runs) < 5:
            print("Not enough data for surrogate model (need at least 5 points)")
            return None, None, None
        
        # Prepare features (parameters) and target (power)
        feature_cols = ['x1_um', 'x2_um', 'x3_um', 'y1_um', 'y2_um', 'y3_um', 'AlNzSpan_um', 'DFTz_um', 'theta_deg']
        X = successful_runs[feature_cols].values
        y = successful_runs['power'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Gaussian Process
        kernel = ConstantKernel(1.0) * RBF(1.0)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        gp.fit(X_scaled, y)
        
        # Predict on a grid around the best points
        best_point = successful_runs.loc[successful_runs['power'].idxmax(), feature_cols].values
        
        print(f"\n" + "="*60)
        print("SURROGATE MODEL ANALYSIS")
        print("="*60)
        print("Fitted Gaussian Process to predict optimal parameters beyond sampled points...")
        
        # Generate predictions around the best point
        n_samples = 1000
        perturbations = np.random.normal(0, 0.1, (n_samples, len(best_point)))
        test_points = best_point + perturbations * best_point  # 10% perturbations
        
        # Scale test points
        test_points_scaled = scaler.transform(test_points)
        
        # Predict
        y_pred, y_std = gp.predict(test_points_scaled, return_std=True)
        
        # Find best predicted point
        best_pred_idx = np.argmax(y_pred)
        best_pred_params = test_points[best_pred_idx]
        best_pred_power = y_pred[best_pred_idx]
        best_pred_uncertainty = y_std[best_pred_idx]
        
        print(f"Best predicted power: {best_pred_power:.6f} ± {best_pred_uncertainty:.6f}")
        print(f"Best sampled power: {successful_runs['power'].max():.6f}")
        
        if best_pred_power > successful_runs['power'].max():
            print("✓ Surrogate model found potentially better parameters!")
            
            print(f"\nPredicted optimal parameters:")
            for i, col in enumerate(feature_cols):
                print(f"  {col}: {best_pred_params[i]:.6e}")
                
            return best_pred_params, best_pred_power, feature_cols
        else:
            print("✗ No improvement found beyond sampled points")
            return None, None, None
            
    except ImportError:
        print("sklearn not available - cannot fit surrogate model")
        return None, None, None
    except Exception as e:
        print(f"Error fitting surrogate model: {e}")
        return None, None, None

if __name__ == "__main__":
    print("="*80)
    print("BAYESIAN OPTIMIZATION RESULT ANALYSIS")
    print("="*80)
    print("This script analyzes your interrupted Bayesian optimization in two ways:")
    print("1. SAMPLED RESULTS: Best from actual tested parameters")
    print("2. SURROGATE MODEL: Predicted optimal parameters using Gaussian Process")
    print("="*80)
    
    # Analyze the results
    best_result = analyze_bayesian_results("final_results.csv")
    
    if best_result is not None:
        # Try to fit surrogate model for potentially better parameters
        df = pd.read_csv("final_results.csv")
        pred_params, pred_power, feature_cols = fit_surrogate_model(df)
        
        # Create a test script with the best parameters
        create_test_script(best_result, "test_best_triangle_params.py")
        
        if pred_params is not None:
            # Create test script for predicted optimal parameters
            pred_result = pd.Series({
                'x1_um': pred_params[0], 'x2_um': pred_params[1], 'x3_um': pred_params[2],
                'y1_um': pred_params[3], 'y2_um': pred_params[4], 'y3_um': pred_params[5],
                'AlNzSpan_um': pred_params[6], 'DFTz_um': pred_params[7], 'theta_deg': pred_params[8],
                'power': pred_power, 'iteration': 'predicted'
            })
            create_test_script(pred_result, "test_predicted_optimal_params.py")
        
        print(f"\n" + "="*60)
        print("SUMMARY: SAMPLED vs PREDICTED OPTIMUM")
        print("="*60)
        print("SAMPLED BEST (from your actual runs):")
        print(f"  Power: {best_result['power']:.6f}")
        print(f"  Iteration: {best_result['iteration']}")
        
        if pred_params is not None:
            print(f"\nPREDICTED OPTIMUM (from surrogate model):")
            print(f"  Power: {pred_power:.6f}")
            print(f"  Status: Untested prediction")
            print(f"\n⚠️  IMPORTANT: Test the predicted parameters to verify!")
        
        print(f"\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. The 'sampled best' is guaranteed to work (it was tested)")
        print("2. The 'predicted optimum' might be better but needs verification")
        print("3. Use both test scripts to compare actual vs predicted performance")
        print("4. Consider running more Bayesian optimization starting from the best point")
