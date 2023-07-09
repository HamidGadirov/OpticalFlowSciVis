import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load PSNR scores from pickle files
with open('psnr_LREST.pkl', 'rb') as f:
    psnr_LREST = pickle.load(f)

with open('psnr_RIFE.pkl', 'rb') as f:
    psnr_RIFE = pickle.load(f)

with open('psnr_base_linear.pkl', 'rb') as f:
    psnr_base_linear = pickle.load(f)

# Load LPIPS scores from pickle files
with open('lpips_LREST.pkl', 'rb') as f:
    lpips_LREST = pickle.load(f)

with open('lpips_RIFE.pkl', 'rb') as f:
    lpips_RIFE = pickle.load(f)

with open('lpips_base_linear.pkl', 'rb') as f:
    lpips_base_linear = pickle.load(f)

# Squeeze the data to remove any single-dimensional entries
psnr_LREST = np.squeeze(psnr_LREST)
psnr_RIFE = np.squeeze(psnr_RIFE)
psnr_base_linear = np.squeeze(psnr_base_linear)
lpips_LREST = np.squeeze(lpips_LREST)
lpips_RIFE = np.squeeze(lpips_RIFE)
lpips_base_linear = np.squeeze(lpips_base_linear)

# Create a figure and axis
fig, ax1 = plt.subplots()
ax1.set_title('PSNR and LPIPS Comparison')
ax1.set_xlabel('Method')
ax1.set_ylabel('PSNR')

# Create the first boxplot for PSNR
bp1 = ax1.boxplot([psnr_LREST, psnr_RIFE, psnr_base_linear], positions=[1, 2, 3], widths=0.25, patch_artist=True,
                  flierprops={'marker': 'd'}, whis=1.5)

# Create a second y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('LPIPS')

# Create the second boxplot for LPIPS
bp2 = ax2.boxplot([lpips_LREST, lpips_RIFE, lpips_base_linear], positions=[1.2, 2.2, 3.2], widths=0.25, patch_artist=True,
                  flierprops={'marker': 'd'}, whis=3)

# Customize the appearance of the boxplots
for box in bp1['boxes']:
    box.set(color='blue', linewidth=1)
    box.set_facecolor('lightblue')
for box in bp2['boxes']:
    box.set(color='purple', linewidth=1)
    box.set_facecolor('mediumpurple')
# for box in bp2['boxes']:
#     box.set(color='green', linewidth=1)
#     box.set_facecolor('lightgreen')

# Set the labels for the x-axis
ax1.set_xticks([1.1, 2.1, 3.1])
ax1.set_xticklabels(['LREST', 'RIFE', 'Linear'])

# Create a legend
legend = [bp1['boxes'][0], bp2['boxes'][0]]
labels = ['PSNR', 'LPIPS']
ax1.legend(legend, labels, loc='upper right')

# Save the plot as an image file
plt.savefig('psnr_lpips_boxplot.pdf')

# Show the plot
plt.show()
