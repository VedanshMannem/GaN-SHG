"""
Advanced plotting script for fixed_BOp.csv data

Column names: iteration, x, y, z, x_span, y_span, z_span, DFTz, angle, power, success
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Define column names
column_names = [
    "iteration", "x", "y", "z", "x_span", "y_span", 
    "z_span", "DFTz", "angle", "power", "success"
]

# Load data
df = pd.read_csv("fixed_BOp.csv", header=None, names=column_names)
print(f"Loaded {len(df)} rows of data")

# Filter for successful runs only
df_success = df[df['success'] == 'success'].copy()
print(f"Found {len(df_success)} successful runs")

# Create plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Rectangle Optimization Data Analysis', fontsize=16, fontweight='bold')

# 1. Power vs Iteration
ax1 = axes[0, 0]
ax1.plot(df_success['iteration'], df_success['power'], 'o-', alpha=0.7, markersize=4)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Power')
ax1.set_title('Power vs Iteration')
ax1.grid(True, alpha=0.3)

# 2. Power vs X Span
ax2 = axes[0, 1]
ax2.scatter(df_success['x_span'], df_success['power'], alpha=0.7, s=30)
ax2.set_xlabel('X Span')
ax2.set_ylabel('Power')
ax2.set_title('Power vs X Span')
ax2.grid(True, alpha=0.3)

# 3. Power vs Y Span
ax3 = axes[0, 2]
ax3.scatter(df_success['y_span'], df_success['power'], alpha=0.7, s=30)
ax3.set_xlabel('Y Span')
ax3.set_ylabel('Power')
ax3.set_title('Power vs Y Span')
ax3.grid(True, alpha=0.3)

# 4. Power vs Z Span
ax4 = axes[1, 0]
ax4.scatter(df_success['z_span'], df_success['power'], alpha=0.7, s=30)
ax4.set_xlabel('Z Span')
ax4.set_ylabel('Power')
ax4.set_title('Power vs Z Span')
ax4.grid(True, alpha=0.3)

# 5. Power vs Angle
ax5 = axes[1, 1]
ax5.scatter(df_success['angle'], df_success['power'], alpha=0.7, s=30)
ax5.set_xlabel('Angle')
ax5.set_ylabel('Power')
ax5.set_title('Power vs Angle')
ax5.grid(True, alpha=0.3)

# 6. Power vs DFTz
ax6 = axes[1, 2]
ax6.scatter(df_success['DFTz'], df_success['power'], alpha=0.7, s=30)
ax6.set_xlabel('DFTz')
ax6.set_ylabel('Power')
ax6.set_title('Power vs DFTz')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Print basic statistics
print(f"\nBasic Statistics:")
print(f"Power range: {df_success['power'].min():.3f} to {df_success['power'].max():.3f}")
print(f"Best power: {df_success['power'].max():.3f} at iteration {df_success.loc[df_success['power'].idxmax(), 'iteration']}")

# Show correlation with power
correlations = df_success[['x_span', 'y_span', 'z_span', 'DFTz', 'angle', 'power']].corr()['power'].sort_values(ascending=False)
print(f"\nCorrelations with power:")
for param, corr in correlations.items():
    if param != 'power':
        print(f"  {param}: {corr:.3f}")

print("\nPlot saved as 'simple_plots.png'")

# ============================================================================
# ADVANCED PLOTS
# ============================================================================

# Set up matplotlib and seaborn style
plt.style.use('default')
sns.set_palette("husl")

# 1. CORRELATION HEATMAP
print("\nCreating correlation heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
param_cols = ['x', 'y', 'z', 'x_span', 'y_span', 'z_span', 'DFTz', 'angle', 'power']
corr_matrix = df_success[param_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title('Parameter Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 3D SCATTER PLOT
print("Creating 3D scatter plot...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Use the three spans as 3D coordinates, color by power
scatter = ax.scatter(df_success['x_span'], df_success['y_span'], df_success['z_span'],
                    c=df_success['power'], cmap='viridis', s=60, alpha=0.8)

ax.set_xlabel('X Span')
ax.set_ylabel('Y Span')
ax.set_zlabel('Z Span')
ax.set_title('3D Parameter Space (colored by Power)', fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
cbar.set_label('Power', rotation=270, labelpad=15)

# Highlight best result
best_idx = df_success['power'].idxmax()
ax.scatter(df_success.loc[best_idx, 'x_span'], 
          df_success.loc[best_idx, 'y_span'], 
          df_success.loc[best_idx, 'z_span'],
          c='red', s=200, marker='*', label='Best Result')
ax.legend()

plt.savefig('3d_parameter_space.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. OPTIMIZATION PROGRESS WITH ROLLING STATISTICS
print("Creating optimization progress analysis...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Optimization Progress Analysis', fontsize=16, fontweight='bold')

# Sort by iteration for proper time series
df_sorted = df_success.sort_values('iteration').reset_index(drop=True)

# Calculate rolling statistics
window_size = 5
df_sorted['power_rolling_mean'] = df_sorted['power'].rolling(window=window_size, min_periods=1).mean()
df_sorted['power_cummax'] = df_sorted['power'].expanding().max()

# Power evolution with trends
ax1 = axes[0, 0]
ax1.plot(df_sorted['iteration'], df_sorted['power'], 'o-', alpha=0.7, label='Actual Power', markersize=4)
ax1.plot(df_sorted['iteration'], df_sorted['power_rolling_mean'], '-', linewidth=2, 
         label=f'Rolling Mean (window={window_size})', color='red')
ax1.plot(df_sorted['iteration'], df_sorted['power_cummax'], '-', linewidth=2, 
         label='Best So Far', color='green')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Power')
ax1.set_title('Power Evolution with Trends')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Power distribution
ax2 = axes[0, 1]
ax2.hist(df_success['power'], bins=15, alpha=0.7, edgecolor='black', color='skyblue')
ax2.axvline(df_success['power'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax2.axvline(df_success['power'].median(), color='orange', linestyle='--', linewidth=2, label='Median')
ax2.set_xlabel('Power')
ax2.set_ylabel('Frequency')
ax2.set_title('Power Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Parameter evolution over iterations
ax3 = axes[1, 0]
params_to_plot = ['angle', 'DFTz']
colors = ['blue', 'red']
for i, param in enumerate(params_to_plot):
    ax3_twin = ax3 if i == 0 else ax3.twinx()
    ax3_twin.plot(df_sorted['iteration'], df_sorted[param], 'o-', 
                  color=colors[i], alpha=0.7, label=param)
    ax3_twin.set_ylabel(param, color=colors[i])
    ax3_twin.tick_params(axis='y', labelcolor=colors[i])

ax3.set_xlabel('Iteration')
ax3.set_title('Key Parameter Evolution')
ax3.grid(True, alpha=0.3)

# Top results analysis
ax4 = axes[1, 1]
top_n = 5
top_results = df_success.nlargest(top_n, 'power')
bars = ax4.bar(range(len(top_results)), top_results['power'], 
               color=plt.cm.viridis(np.linspace(0, 1, len(top_results))), alpha=0.8)
ax4.set_xlabel('Rank')
ax4.set_ylabel('Power')
ax4.set_title(f'Top {top_n} Results')
ax4.set_xticks(range(len(top_results)))
ax4.set_xticklabels([f'#{i+1}' for i in range(len(top_results))])

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom')

ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimization_progress.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. PAIRWISE PARAMETER RELATIONSHIPS
print("Creating pairwise parameter analysis...")
# Select most important parameters
key_params = ['x_span', 'y_span', 'z_span', 'angle', 'power']
g = sns.PairGrid(df_success[key_params], diag_sharey=False)
g.map_upper(sns.scatterplot, alpha=0.7)
g.map_lower(sns.scatterplot, alpha=0.7)
g.map_diag(sns.histplot, kde=True)

# Add correlation coefficients to upper triangle
for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    corr = df_success[key_params].iloc[:, [i, j]].corr().iloc[0, 1]
    g.axes[i, j].text(0.5, 0.5, f'r = {corr:.3f}', 
                     transform=g.axes[i, j].transAxes, 
                     ha='center', va='center', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

g.fig.suptitle('Pairwise Parameter Relationships', fontsize=16, fontweight='bold', y=1.02)
plt.savefig('pairwise_relationships.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. PARAMETER SENSITIVITY ANALYSIS
print("Creating parameter sensitivity analysis...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')

param_list = ['x_span', 'y_span', 'z_span', 'DFTz', 'angle']
param_labels = ['X Span', 'Y Span', 'Z Span', 'DFTz', 'Angle']

for i, (param, label) in enumerate(zip(param_list, param_labels)):
    if i < 5:
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Create binned analysis
        df_temp = df_success.copy()
        df_temp['param_bin'] = pd.cut(df_temp[param], bins=5, labels=['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])
        
        # Box plot
        sns.boxplot(data=df_temp, x='param_bin', y='power', ax=ax)
        ax.set_xlabel(f'{label} (Binned)')
        ax.set_ylabel('Power')
        ax.set_title(f'Power Distribution by {label} Range')
        ax.tick_params(axis='x', rotation=45)
        
        # Add trend line on scatter
        ax_twin = ax.twinx()
        ax_twin.scatter(df_success[param], df_success['power'], alpha=0.5, color='red', s=20)
        
        # Fit polynomial trend
        z = np.polyfit(df_success[param], df_success['power'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(df_success[param].min(), df_success[param].max(), 100)
        ax_twin.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        ax_twin.set_ylabel('Power (Scatter)', color='red')
        ax_twin.tick_params(axis='y', labelcolor='red')

# Remove empty subplot
axes[1, 2].remove()

plt.tight_layout()
plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. MULTI-DIMENSIONAL ANALYSIS
print("Creating multi-dimensional analysis...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Multi-Dimensional Parameter Analysis', fontsize=16, fontweight='bold')

# Bubble chart: x_span vs y_span, size=z_span, color=power
ax1 = axes[0, 0]
scatter = ax1.scatter(df_success['x_span'], df_success['y_span'], 
                     s=df_success['z_span']*1000, c=df_success['power'], 
                     cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
ax1.set_xlabel('X Span')
ax1.set_ylabel('Y Span')
ax1.set_title('X vs Y Span (size=Z Span, color=Power)')
plt.colorbar(scatter, ax=ax1, label='Power')

# Angle vs DFTz colored by power
ax2 = axes[0, 1]
scatter2 = ax2.scatter(df_success['angle'], df_success['DFTz'], 
                      c=df_success['power'], cmap='plasma', s=60, alpha=0.8)
ax2.set_xlabel('Angle')
ax2.set_ylabel('DFTz')
ax2.set_title('Angle vs DFTz (colored by Power)')
plt.colorbar(scatter2, ax=ax2, label='Power')

# Performance over parameter space
ax3 = axes[1, 0]
# Create 2D histogram
h = ax3.hist2d(df_success['x_span'], df_success['y_span'], 
               weights=df_success['power'], bins=8, cmap='YlOrRd')
ax3.set_xlabel('X Span')
ax3.set_ylabel('Y Span')
ax3.set_title('Power Density in X-Y Span Space')
plt.colorbar(h[3], ax=ax3, label='Average Power')

# Parameter importance radar chart
ax4 = axes[1, 1]
# Calculate importance as absolute correlation with power
importance = abs(correlations[correlations.index != 'power'])
params = importance.index.tolist()
values = importance.values.tolist()

# Create radar chart
angles = np.linspace(0, 2*np.pi, len(params), endpoint=False).tolist()
values += values[:1]  # Complete the circle
angles += angles[:1]

ax4.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
ax4.fill(angles, values, alpha=0.25, color='blue')
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(params)
ax4.set_ylim(0, max(values))
ax4.set_title('Parameter Importance\n(Absolute Correlation with Power)')
ax4.grid(True)

plt.tight_layout()
plt.savefig('multidimensional_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# SUMMARY STATISTICS
print("\n" + "="*60)
print("COMPREHENSIVE ANALYSIS SUMMARY")
print("="*60)

print(f"Dataset: {len(df_success)} successful optimization runs")
print(f"Power range: {df_success['power'].min():.3f} to {df_success['power'].max():.3f}")
print(f"Power improvement: {((df_success['power'].max() - df_success['power'].min()) / df_success['power'].min() * 100):.1f}%")

print(f"\nBest Result (Iteration {df_success.loc[df_success['power'].idxmax(), 'iteration']}):")
print(f"  Power: {df_success['power'].max():.3f}")
best_idx = df_success['power'].idxmax()
for param in ['x_span', 'y_span', 'z_span', 'DFTz', 'angle']:
    print(f"  {param}: {df_success.loc[best_idx, param]:.6f}")

print(f"\nParameter Statistics:")
for param in ['x_span', 'y_span', 'z_span', 'DFTz', 'angle']:
    mean_val = df_success[param].mean()
    std_val = df_success[param].std()
    range_val = df_success[param].max() - df_success[param].min()
    print(f"  {param}: mean={mean_val:.6f}, std={std_val:.6f}, range={range_val:.6f}")

print(f"\nPlots saved:")
print("  - simple_plots.png (basic parameter relationships)")
print("  - correlation_heatmap.png (parameter correlations)")
print("  - 3d_parameter_space.png (3D visualization)")
print("  - optimization_progress.png (optimization trends)")
print("  - pairwise_relationships.png (detailed pair analysis)")
print("  - parameter_sensitivity.png (sensitivity analysis)")
print("  - multidimensional_analysis.png (advanced visualizations)")
