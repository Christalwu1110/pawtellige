import matplotlib.pyplot as plt
import numpy as np

# Set up the plot style for academic posters
plt.style.use('default')  # Reset to default style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# Create data
models = ['ResNet-50', 'EfficientNet-B0']
parameters = [25.6, 5.3]  # Parameters (Millions)
accuracy = [76.0, 77.1]   # Accuracy (%), replace with your actual data

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Set positions and width for bars
x = np.arange(len(models))
width = 0.35

# Plot parameters (left Y-axis)
bars1 = ax1.bar(x - width/2, parameters, width, 
                color='#4C72B0', edgecolor='black', 
                label='Parameters (M)', alpha=0.8)

# Create second Y-axis for accuracy
ax2 = ax1.twinx()

# Plot accuracy (right Y-axis)
bars2 = ax2.bar(x + width/2, accuracy, width, 
                color='#DD8452', edgecolor='black', 
                label='Accuracy (%)', alpha=0.8)

# Set labels and title
ax1.set_xlabel('Model Architecture', fontweight='bold')
ax1.set_ylabel('Parameters (Millions)', fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontweight='bold')
ax1.set_title('Model Comparison: Efficiency vs. Performance', fontsize=16, fontweight='bold', pad=20)

# Set X-axis ticks
ax1.set_xticks(x)
ax1.set_xticklabels(models)

# Add value labels on bars
def add_value_labels(bars, axis, offset=0.05):
    for bar in bars:
        height = bar.get_height()
        axis.text(bar.get_x() + bar.get_width()/2., height + offset,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=11)

add_value_labels(bars1, ax1)
add_value_labels(bars2, ax2)

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)

# Adjust layout
plt.tight_layout()

# Save high-resolution image
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('model_comparison.pdf', bbox_inches='tight')  # PDF for vector graphics

# Show plot
plt.show()