import pickle
import matplotlib.pyplot as plt
import numpy as np

# Define custom colors for each model (lighter shades)
colors = {
    'LREST': '#6699cc',
    'RIFE': '#ff6666',
    'Linear': '#99cc99'
}

# Load and process PSNR data for each model and factor
data = {}
models = ['LREST', 'RIFE', 'Linear']
# factors = ['2x', '3x', '4x', '5x', '8x', '12x', '16x', '24x', '32x', '48x', '64x', '96x', '128x']
factors = ['3x', '5x', '8x', '12x', '16x', '32x', '48x', '64x']

for model in models:
    data[model] = []
    for factor in factors:
        filename = f'psnr_{model}_{factor}.pkl'
        with open(filename, 'rb') as f:
            psnr = np.squeeze(pickle.load(f))
            data[model].append(psnr)

# Create a figure and axis
fig, ax1 = plt.subplots()
ax1.set_title('Methods Comparison, LBS')
ax1.set_xlabel('Factor')
ax1.set_ylabel('PSNR score')

# Create boxplots for each model's data
boxplots = []
positions = np.arange(len(factors))
for i, model in enumerate(models):
    bp = ax1.boxplot(data[model], positions=positions + i * 0.25, widths=0.25, patch_artist=True, showfliers=False)
    boxplots.append(bp)

# Customize the appearance of the boxplots
for i, model in enumerate(models):
    for box in boxplots[i]['boxes']:
        box.set(color='black', linewidth=1)
        box.set_facecolor(colors[model])

# Set the labels for the x-axis
ax1.set_xticks(positions + 0.25)
ax1.set_xticklabels(factors)

# Create a legend
legend_labels = [plt.Rectangle((0, 0), 1, 1, color=colors[model], label=model) for model in models]
ax1.legend(handles=legend_labels, loc='upper right')

# Save the plot as an image file
plt.savefig('psnr_boxplot_lbs.pdf')

# Show the plot
plt.show()
