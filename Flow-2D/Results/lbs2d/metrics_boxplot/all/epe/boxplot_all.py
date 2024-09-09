import pickle
import matplotlib.pyplot as plt
import numpy as np

# Define custom colors for each model (lighter shades)
colors = {
    'FLINT': '#6699cc',
    'RIFE': '#ff6666',
    'Linear': '#99cc99'
}

# Load and process epe data for each model and factor
data = {}
models = ['FLINT', 'RIFE']
factors = ['3x', '5x', '8x', '12x', '16x', '32x', '48x', '64x']

for model in models:
    if model == "FLINT": 
        model = "LREST"
    data[model] = []
    for factor in factors:
        filename = f'epe_{model}_{factor}.pkl'
        with open(filename, 'rb') as f:
            epe = np.squeeze(pickle.load(f))
            data[model].append(epe)

# Create a figure and axis
fig, ax1 = plt.subplots()

# Increase font size for all text elements
font_size = 14
plt.rcParams.update({'font.size': font_size})

# Create boxplots for each model's data
boxplots = []
positions = np.arange(len(factors))
for i, model in enumerate(models):
    if model == "FLINT": 
        model = "LREST"
    bp = ax1.boxplot(data[model], positions=positions + i * 0.25, widths=0.25, patch_artist=True, showfliers=False)
    boxplots.append(bp)

# Customize the appearance of the boxplots
for i, model in enumerate(models):
    for box in boxplots[i]['boxes']:
        box.set(color='black', linewidth=1)
        box.set_facecolor(colors[model])
    
    # Change median line color to black
    for median in boxplots[i]['medians']:
        median.set_color('black')

# Set the labels for the x-axis
ax1.set_xticks(positions + 0.25)
ax1.set_xticklabels(factors)

# Increase font size for axis labels and numbers on the axes
ax1.tick_params(axis='both', which='both', labelsize=font_size)

# Create a legend with increased font size
legend_labels = [plt.Rectangle((0, 0), 1, 1, color=colors[model], label=model) for model in models]
ax1.legend(handles=legend_labels, loc='upper right', fontsize=font_size)

# Save the plot as an image file
plt.savefig('epe_boxplot_lbs_new.pdf')

# Show the plot
plt.show()
