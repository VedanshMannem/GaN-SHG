import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# import seaborn as sns
import os
import warnings

class RectangleRandomForest:
 
    def __init__(self, data_dir="./"):
        self.data_dir = data_dir
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.combined_data = None
        
    def load_and_combine_data(self):
        """
        Load and combine data from both CSV files
        """
        # Load rectBOp_data.csv
        rectBOp_file = os.path.join(self.data_dir, "rectBOp_data.csv")
        second_bay_file = os.path.join(self.data_dir, "second_bayesian_optimization.csv")
        
        print("Loading data files...")
        
        # Load first dataset
        if os.path.exists(rectBOp_file):
            df1 = pd.read_csv(rectBOp_file)
            print(f"Loaded {len(df1)} samples from rectBOp_data.csv")
            
            # Convert power column to numeric, handling any string values
            df1['power'] = pd.to_numeric(df1['power'], errors='coerce')
            
            # Filter successful runs only
            df1_success = df1[(df1['status'] == 'success') & (df1['power'].notna())].copy()
            print(f"Found {len(df1_success)} successful runs in rectBOp_data.csv")
            
            # Rename columns for consistency and add source
            df1_clean = df1_success[['x', 'y', 'z', 'xspan', 'yspan', 'zspan', 'power']].copy()
            df1_clean['source'] = 'rectBOp'
            
        else:
            print(f"Warning: {rectBOp_file} not found")
            df1_clean = pd.DataFrame()
        
        # Load second dataset
        if os.path.exists(second_bay_file):
            df2 = pd.read_csv(second_bay_file)
            print(f"Loaded {len(df2)} samples from second_bayesian_optimization.csv")
            
            # Convert power column to numeric, handling any string values
            df2['power'] = pd.to_numeric(df2['power'], errors='coerce')
            
            # Filter successful runs only (exclude historical with power=0 and NaN values)
            df2_success = df2[(df2['status'] == 'success') & (df2['power'].notna()) & (df2['power'] > 0)].copy()
            print(f"Found {len(df2_success)} valid runs in second_bayesian_optimization.csv")
            
            # Rename columns for consistency and add source
            df2_clean = df2_success[['AlNxSpan_um', 'AlNySpan_um', 'AlNzSpan_um', 
                                   'DFTz_um', 'theta_deg', 'power']].copy()
            df2_clean.columns = ['xspan', 'yspan', 'zspan', 'z_offset', 'theta', 'power']
            df2_clean['source'] = 'second_bay'
            
            # Add missing columns with default values for compatibility
            df2_clean['x'] = 0.0  # Position parameters not in second dataset
            df2_clean['y'] = 0.0
            df2_clean['z'] = df2_clean['z_offset']  # Use DFTz as z position
            
        else:
            print(f"Warning: {second_bay_file} not found")
            df2_clean = pd.DataFrame()
        
        # Combine datasets
        if not df1_clean.empty and not df2_clean.empty:
            # Align columns
            common_cols = ['x', 'y', 'z', 'xspan', 'yspan', 'zspan', 'power', 'source']
            
            # Add theta column to df1 (default angle)
            df1_clean['theta'] = 45.0  # Default theta value
            df1_clean['z_offset'] = df1_clean['z']  # Use z as z_offset
            
            # Add z_offset to final columns
            common_cols.extend(['theta', 'z_offset'])
            
            # Ensure both dataframes have all columns
            for col in common_cols:
                if col not in df1_clean.columns:
                    df1_clean[col] = 0.0
                if col not in df2_clean.columns:
                    df2_clean[col] = 0.0
            
            self.combined_data = pd.concat([df1_clean[common_cols], df2_clean[common_cols]], 
                                         ignore_index=True)
        elif not df1_clean.empty:
            df1_clean['theta'] = 45.0
            df1_clean['z_offset'] = df1_clean['z']
            self.combined_data = df1_clean
        elif not df2_clean.empty:
            self.combined_data = df2_clean
        else:
            raise ValueError("No valid data found in either CSV file")
        
        # Remove any rows with infinite or NaN values
        initial_len = len(self.combined_data)
        
        # Convert all numeric columns to numeric type
        numeric_cols = ['x', 'y', 'z', 'xspan', 'yspan', 'zspan', 'power', 'theta', 'z_offset']
        for col in numeric_cols:
            if col in self.combined_data.columns:
                self.combined_data[col] = pd.to_numeric(self.combined_data[col], errors='coerce')
        
        self.combined_data = self.combined_data.replace([np.inf, -np.inf], np.nan).dropna()
        final_len = len(self.combined_data)
        
        if initial_len != final_len:
            print(f"Removed {initial_len - final_len} rows with invalid values")
        
        print(f"Combined dataset: {len(self.combined_data)} total samples")
        print(f"Power range: {self.combined_data['power'].min():.2e} to {self.combined_data['power'].max():.2e}")
        
        return self.combined_data
    
    def prepare_features(self, log_transform_power=True):

        if self.combined_data is None:
            raise ValueError("Data not loaded. Call load_and_combine_data() first.")
        
        self.feature_names = ['x', 'y', 'z', 'xspan', 'yspan', 'zspan', 'theta', 'z_offset']
        
        X = self.combined_data[self.feature_names].copy()
        
        y = self.combined_data['power'].copy()
        
        if log_transform_power:
            # Add small constant to avoid log(0)
            y_min = y[y > 0].min() if (y > 0).any() else 1e-20
            y_safe = np.where(y <= 0, y_min/10, y)
            y = np.log10(y_safe)
            print(f"Applied log10 transformation to power values")
            print(f"Log power range: {y.min():.2f} to {y.max():.2f}")
        
        self.log_transformed = log_transform_power
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42, optimize_hyperparameters=True):
        """
        Train the Random Forest model
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            optimize_hyperparameters: Whether to perform hyperparameter optimization
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        if optimize_hyperparameters:
            print("Optimizing hyperparameters...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Create base model
            rf_base = RandomForestRegressor(random_state=random_state)
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                rf_base, param_grid, 
                cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # Use best model
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            print(f"Best parameters: {best_params}")
            
        else:
            # Use default parameters with some optimization
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print("\n=== Model Performance ===")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, 
                                   cv=5, scoring='r2')
        print(f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Store results
        self.results = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_train_pred': y_train_pred, 'y_test_pred': y_test_pred,
            'train_mse': train_mse, 'test_mse': test_mse,
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_mae': train_mae, 'test_mae': test_mae,
            'cv_scores': cv_scores
        }
        
        return self.results
    
    def plot_results(self, save_plots=True):
        """
        Create visualization plots for the model results
        
        Args:
            save_plots (bool): Whether to save plots to files
        """
        if self.results is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Set up the plotting style
        plt.style.use('default')
        
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Random Forest Model Results', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted (Training)
        ax1 = axes[0, 0]
        ax1.scatter(self.results['y_train'], self.results['y_train_pred'], 
                   alpha=0.6, color='blue', label='Training')
        ax1.plot([self.results['y_train'].min(), self.results['y_train'].max()], 
                [self.results['y_train'].min(), self.results['y_train'].max()], 
                'r--', lw=2)
        ax1.set_xlabel('Actual Power')
        ax1.set_ylabel('Predicted Power')
        ax1.set_title(f'Training Set (R² = {self.results["train_r2"]:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Actual vs Predicted (Test)
        ax2 = axes[0, 1]
        ax2.scatter(self.results['y_test'], self.results['y_test_pred'], 
                   alpha=0.6, color='green', label='Test')
        ax2.plot([self.results['y_test'].min(), self.results['y_test'].max()], 
                [self.results['y_test'].min(), self.results['y_test'].max()], 
                'r--', lw=2)
        ax2.set_xlabel('Actual Power')
        ax2.set_ylabel('Predicted Power')
        ax2.set_title(f'Test Set (R² = {self.results["test_r2"]:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature Importance
        ax3 = axes[1, 0]
        feature_importance = self.model.feature_importances_
        indices = np.argsort(feature_importance)[::-1]
        
        ax3.bar(range(len(feature_importance)), feature_importance[indices])
        ax3.set_xlabel('Features')
        ax3.set_ylabel('Importance')
        ax3.set_title('Feature Importance')
        ax3.set_xticks(range(len(feature_importance)))
        ax3.set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuals Plot
        ax4 = axes[1, 1]
        residuals_train = self.results['y_train'] - self.results['y_train_pred']
        residuals_test = self.results['y_test'] - self.results['y_test_pred']
        
        ax4.scatter(self.results['y_train_pred'], residuals_train, 
                   alpha=0.6, color='blue', label='Training')
        ax4.scatter(self.results['y_test_pred'], residuals_test, 
                   alpha=0.6, color='green', label='Test')
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Predicted Power')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals Plot')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('random_forest_results.png', dpi=300, bbox_inches='tight')
            print("Plots saved as 'random_forest_results.png'")
        
        plt.show()
        
        # Feature importance summary
        print("\n=== Feature Importance ===")
        for i in indices:
            print(f"{self.feature_names[i]}: {feature_importance[i]:.4f}")
    
    def predict(self, X_new):
        """
        Make predictions on new data
        
        Args:
            X_new: New feature matrix
            
        Returns:
            Predictions (transformed back if log transformation was used)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Ensure X_new has the correct columns
        if isinstance(X_new, dict):
            X_new = pd.DataFrame([X_new])
        
        # Ensure all feature columns are present
        for col in self.feature_names:
            if col not in X_new.columns:
                X_new[col] = 0.0  # Default value
        
        X_new_scaled = self.scaler.transform(X_new[self.feature_names])
        predictions = self.model.predict(X_new_scaled)
        
        # Transform back if log transformation was used
        if hasattr(self, 'log_transformed') and self.log_transformed:
            predictions = 10 ** predictions
        
        return predictions
    
    def optimize_parameters(self, target_power=None, n_suggestions=10):
        """
        Use the trained model to suggest optimal parameters
        
        Args:
            target_power: Target power value (optional)
            n_suggestions: Number of parameter suggestions to generate
            
        Returns:
            DataFrame with suggested parameters and predicted powers
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Generate random parameter combinations within reasonable bounds
        np.random.seed(42)
        
        # Define parameter ranges based on training data
        param_ranges = {}
        for feature in self.feature_names:
            feature_data = self.combined_data[feature]
            # Convert to numeric if not already
            feature_data = pd.to_numeric(feature_data, errors='coerce')
            # Remove any NaN values
            feature_data = feature_data.dropna()
            param_ranges[feature] = (feature_data.min(), feature_data.max())
        
        suggestions = []
        
        for _ in range(n_suggestions * 10):  # Generate more to filter best ones
            suggestion = {}
            for feature in self.feature_names:
                min_val, max_val = param_ranges[feature]
                suggestion[feature] = np.random.uniform(min_val, max_val)
            
            # Predict power for this combination
            pred_power = self.predict(pd.DataFrame([suggestion]))[0]
            suggestion['predicted_power'] = pred_power
            suggestions.append(suggestion)
        
        # Convert to DataFrame and sort by predicted power
        suggestions_df = pd.DataFrame(suggestions)
        suggestions_df = suggestions_df.sort_values('predicted_power', ascending=False)
        
        print(f"\n=== Top {n_suggestions} Parameter Suggestions ===")
        for i, (_, row) in enumerate(suggestions_df.head(n_suggestions).iterrows()):
            print(f"\nSuggestion {i+1}:")
            for feature in self.feature_names:
                print(f"  {feature}: {row[feature]:.4f}")
            print(f"  Predicted Power: {row['predicted_power']:.2e}")
        
        return suggestions_df.head(n_suggestions)


def main():
    """
    Main function to run the Random Forest analysis
    """
    print("=== Rectangle Photonic Structure Random Forest Analysis ===\n")
    
    # Initialize the model
    rf_model = RectangleRandomForest(data_dir="./")
    
    try:
        # Load and combine data
        combined_data = rf_model.load_and_combine_data()
        
        # Display data summary
        print(f"\n=== Data Summary ===")
        print(f"Total samples: {len(combined_data)}")
        print(f"Data sources: {combined_data['source'].value_counts().to_dict()}")
        print(f"\nPower statistics:")
        print(combined_data['power'].describe())
        
        # Prepare features
        X, y = rf_model.prepare_features(log_transform_power=True)
        
        # Train model
        print(f"\n=== Training Random Forest Model ===")
        results = rf_model.train_model(X, y, optimize_hyperparameters=False)
        
        # Create visualizations
        rf_model.plot_results(save_plots=True)
        
        # Generate optimization suggestions
        suggestions = rf_model.optimize_parameters(n_suggestions=5)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Model trained successfully with R² = {results['test_r2']:.4f}")
        print(f"Results and plots saved to current directory")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
