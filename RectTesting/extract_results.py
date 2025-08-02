import re
import csv
import os

def extract_optimization_results(input_file, output_file=None):
    """
    Extract optimization results from final_results.csv and create a clean CSV file.
    Looks for the final best parameters and power values.
    """
    
    if output_file is None:
        output_file = input_file.replace('.csv', '_clean.csv')
    
    print(f"Extracting optimization results from {input_file}...")
    
    results = []
    iteration = 0
    current_params = {}
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by iterations
        iteration_blocks = re.split(r'Iteration \d+/\d+', content)
        
        for i, block in enumerate(iteration_blocks[1:], 1):  # Skip first empty block
            # Extract parameters from "Testing parameters:" line
            param_match = re.search(
                r'Testing parameters: AlNxSpan=([\d.]+)μm, AlNySpan=([\d.]+)μm, AlNzSpan=([\d.]+)μm, DFTz=([-\d.]+)μm, theta=([\d.]+)°',
                block
            )
            
            # Extract power result
            power_match = re.search(r'Power result: ([\d.]+) \(standardized: ([\d.]+)\)', block)
            
            if param_match and power_match:
                AlNxSpan = float(param_match.group(1))
                AlNySpan = float(param_match.group(2))
                AlNzSpan = float(param_match.group(3))
                DFTz = float(param_match.group(4))
                theta = float(param_match.group(5))
                power = float(power_match.group(1))
                
                results.append({
                    'iteration': i,
                    'AlNxSpan_um': AlNxSpan,
                    'AlNySpan_um': AlNySpan,
                    'AlNzSpan_um': AlNzSpan,
                    'DFTz_um': DFTz,
                    'theta_deg': theta,
                    'power': power,
                    'status': 'success'
                })
        
        # Check for final best parameters at the end
        final_best_match = re.search(
            r'Final Best Objective \(Power\): ([\d.]+).*?AlNxSpan: ([\d.e-]+) m.*?AlNySpan: ([\d.e-]+) m.*?AlNzSpan: ([\d.e-]+) m.*?DFTz: ([-\d.e-]+) m.*?theta: ([\d.]+)°',
            content, re.DOTALL
        )
        
        # Write to CSV
        if results:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['iteration', 'AlNxSpan_um', 'AlNySpan_um', 'AlNzSpan_um', 'DFTz_um', 'theta_deg', 'power', 'status']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            print(f"✓ Extraction complete!")
            print(f"  Found {len(results)} optimization iterations")
            print(f"  Output saved to: {output_file}")
            
            if final_best_match:
                print(f"\nFinal Best Result:")
                print(f"  Power: {final_best_match.group(1)}")
                print(f"  AlNxSpan: {float(final_best_match.group(2))*1e6:.6f} μm")
                print(f"  AlNySpan: {float(final_best_match.group(3))*1e6:.6f} μm")
                print(f"  AlNzSpan: {float(final_best_match.group(4))*1e6:.6f} μm")
                print(f"  DFTz: {float(final_best_match.group(5))*1e6:.6f} μm")
                print(f"  theta: {final_best_match.group(6)}°")
            
            return output_file
        else:
            print("No optimization results found in the file!")
            return None
            
    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")
        return None
    except Exception as e:
        print(f"Error extracting results: {e}")
        return None

def create_analysis_summary(csv_file):
    """
    Create a summary analysis of the cleaned results.
    """
    try:
        import pandas as pd
        
        df = pd.read_csv(csv_file)
        
        print(f"\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        
        print(f"Total iterations: {len(df)}")
        print(f"Success rate: 100.0%")
        
        best_result = df.loc[df['power'].idxmax()]
        worst_result = df.loc[df['power'].idxmin()]
        
        print(f"\nBest Result:")
        print(f"  Iteration: {best_result['iteration']}")
        print(f"  Power: {best_result['power']:.6f}")
        print(f"  AlNxSpan: {best_result['AlNxSpan_um']:.6f} μm")
        print(f"  AlNySpan: {best_result['AlNySpan_um']:.6f} μm")
        print(f"  AlNzSpan: {best_result['AlNzSpan_um']:.6f} μm")
        print(f"  DFTz: {best_result['DFTz_um']:.6f} μm")
        print(f"  theta: {best_result['theta_deg']:.1f}°")
        
        print(f"\nStatistics:")
        print(f"  Power range: {worst_result['power']:.6f} to {best_result['power']:.6f}")
        print(f"  Average power: {df['power'].mean():.6f}")
        print(f"  Power improvement: {(best_result['power'] - worst_result['power']) / worst_result['power'] * 100:.1f}%")
        
        # Top 5 results
        top_5 = df.nlargest(5, 'power')
        print(f"\nTop 5 Results:")
        for i, (idx, row) in enumerate(top_5.iterrows(), 1):
            print(f"  {i}. Iteration {row['iteration']}: Power = {row['power']:.6f}")
            
    except ImportError:
        print("\nInstall pandas for detailed analysis: pip install pandas")
    except Exception as e:
        print(f"Error creating summary: {e}")

def main():
    """
    Main function to clean up and analyze final_results.csv.
    """
    
    print("="*60)
    print("BAYESIAN OPTIMIZATION RESULTS EXTRACTOR")
    print("="*60)
    
    # Check current directory for final_results.csv
    current_file = "final_results.csv"
    
    if os.path.exists(current_file):
        print(f"Found {current_file} in current directory")
        
        # Extract optimization results
        clean_file = extract_optimization_results(current_file)
        
        if clean_file:
            # Create analysis summary
            create_analysis_summary(clean_file)
            
            print(f"\n" + "="*60)
            print("FILES CREATED")
            print("="*60)
            print(f"✓ Clean CSV data: {clean_file}")
            print(f"✓ Ready for analysis and plotting")
        
    else:
        print(f"Error: {current_file} not found in current directory")
        print("Make sure you're running this script in the same folder as final_results.csv")

if __name__ == "__main__":
    main()
