import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Define a custom colormap that goes from blue (-1.0) to white (0.0) to red (1.0)
colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
cmap = LinearSegmentedColormap.from_list('my_colormap', colors, N=256)

# Create a sample gradient for the color map bar from -1.0 to 1.0
gradient = np.linspace(-1.0, 1.0, 256).reshape(1, -1)

# Create a figure and add the color map bar to it
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cax = ax.matshow(gradient, cmap=cmap, aspect='auto')
ax.set_xticks([])
ax.set_yticks([])

# Add color bar
cbar = fig.colorbar(cax, orientation='horizontal', ticks=[-1, 0, 1])
cbar.set_label('Color Map Bar')

# Save the color map bar as an image
plt.savefig('color_map_bar.png', bbox_inches='tight', dpi=300)

# Show the plot (optional)
plt.show()
