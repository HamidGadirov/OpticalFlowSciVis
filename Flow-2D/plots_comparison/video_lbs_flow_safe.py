import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import pyimof
import os

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def generate_frame(data, cmap, vmin, vmax):
    norm = Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    frame = mapper.to_rgba(data)[:, :, :3]  # Exclude alpha channel
    frame = np.uint8(frame * 255)  # Convert to uint8
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV
    return frame

# Ensure the output directory exists
output_dir = "LBS_Flow_GT_Pred_Diff"
os.makedirs(output_dir, exist_ok=True)

# Load flow data
flow_gt = load_data('flow_gt.pkl')
flow_pred_LREST = load_data('flow_pred_LREST.pkl')
flow_diff_LREST = load_data('flow_diff_LREST.pkl')
flow_pred_RIFE = load_data('flow_pred_RIFE.pkl')
flow_diff_RIFE = load_data('flow_diff_RIFE.pkl')

# Just to make it faster
# flow_gt = flow_gt[:100]

# Flow Data Visualization
quiver_steps = 9
max_flow_gt = np.max([np.max(np.sqrt(u**2 + v**2)) for u, v in zip(flow_gt[:, 0, ...], flow_gt[:, 1, ...])])

# Set the figure size and dpi to match the desired output frame size
frame_width = 400
frame_height = 100
dpi = 100
fig_size = (frame_width / dpi, frame_height * 5 / dpi)

# Thickness of the separator lines
line_thickness = 2

# Create and save quiver plots
for i in range(len(flow_gt)):
    fig, axs = plt.subplots(nrows=5, figsize=fig_size, dpi=dpi, gridspec_kw={'hspace': 0, 'wspace': 0})

    # GT Flow
    u_gt = flow_gt[i, 0, ...]
    v_gt = flow_gt[i, 1, ...]
    norm_gt = np.sqrt(u_gt * u_gt + v_gt * v_gt)
    img = np.zeros(u_gt.shape)
    pyimof.display.quiver(u_gt, v_gt, c=norm_gt, bg=img, ax=axs[0], cmap='turbo', clim=(0, max_flow_gt), scale=None, bg_cmap='gray', step=quiver_steps)
    axs[0].axis('off')
    # axs[0].text(0.98, 0.05, 'Flow GT', ha='right', va='bottom', transform=axs[0].transAxes, color='red', fontsize=10, weight='bold')
    # axs[0].text(0.98, 0.05, 'Flow GT', ha='right', va='bottom', transform=axs[0].transAxes, color='white', fontdict={'family': 'serif', 'size': 10})

    # LREST Predicted Flow
    u_pred_LREST = flow_pred_LREST[i, 0, ...]
    v_pred_LREST = flow_pred_LREST[i, 1, ...]
    norm_pred_LREST = np.sqrt(u_pred_LREST * u_pred_LREST + v_pred_LREST * v_pred_LREST)
    pyimof.display.quiver(u_pred_LREST, v_pred_LREST, c=norm_pred_LREST, bg=img, ax=axs[1], cmap='turbo', clim=(0, max_flow_gt), scale=None, bg_cmap='gray', step=quiver_steps)
    axs[1].axis('off')
    # axs[1].text(0.98, 0.05, 'Flow FLINT', ha='right', va='bottom', transform=axs[1].transAxes, color='red', fontsize=10, weight='bold')
    # axs[1].text(0.98, 0.05, 'Flow FLINT', ha='right', va='bottom', transform=axs[1].transAxes, color='white', fontdict={'family': 'serif', 'size': 10})


    # LREST Difference Flow
    u_diff_LREST = flow_diff_LREST[i, 0, ...]
    v_diff_LREST = flow_diff_LREST[i, 1, ...]
    norm_diff_LREST = np.sqrt(u_diff_LREST * u_diff_LREST + v_diff_LREST * v_diff_LREST)
    norm_diff_LREST = np.clip(norm_diff_LREST, 0, 0.01)
    pyimof.display.quiver(u_diff_LREST, v_diff_LREST, c=norm_diff_LREST, bg=img, ax=axs[2], cmap='turbo', clim=(0, max_flow_gt), scale=0.001, bg_cmap='gray', step=quiver_steps)
    axs[2].axis('off')
    # axs[2].text(0.98, 0.05, 'Flow diff', ha='right', va='bottom', transform=axs[2].transAxes, color='red', fontsize=10, weight='bold')
    # axs[2].text(0.98, 0.05, 'Flow diff', ha='right', va='bottom', transform=axs[2].transAxes, color='white', fontdict={'family': 'serif', 'size': 10})


    # RIFE Predicted Flow
    u_pred_RIFE = flow_pred_RIFE[i, 0, ...]
    v_pred_RIFE = flow_pred_RIFE[i, 1, ...]
    norm_pred_RIFE = np.sqrt(u_pred_RIFE * u_pred_RIFE + v_pred_RIFE * v_pred_RIFE)
    pyimof.display.quiver(u_pred_RIFE, v_pred_RIFE, c=norm_pred_RIFE, bg=img, ax=axs[3], cmap='turbo', clim=(0, max_flow_gt), scale=None, bg_cmap='gray', step=quiver_steps)
    axs[3].axis('off')
    # axs[3].text(0.98, 0.05, 'Flow RIFE', ha='right', va='bottom', transform=axs[3].transAxes, color='red', fontsize=10, weight='bold')
    # axs[3].text(0.98, 0.05, 'Flow RIFE', ha='right', va='bottom', transform=axs[3].transAxes, color='white', fontdict={'family': 'serif', 'size': 10})


    # RIFE Difference Flow
    u_diff_RIFE = flow_diff_RIFE[i, 0, ...]
    v_diff_RIFE = flow_diff_RIFE[i, 1, ...]
    norm_diff_RIFE = np.sqrt(u_diff_RIFE * u_diff_RIFE + v_diff_RIFE * v_diff_RIFE)
    pyimof.display.quiver(u_diff_RIFE, v_diff_RIFE, c=norm_diff_RIFE, bg=img, ax=axs[4], cmap='turbo', clim=(0, max_flow_gt), scale=None, bg_cmap='gray', step=quiver_steps)
    axs[4].axis('off')
    # axs[4].text(0.98, 0.05, 'Flow diff RIFE', ha='right', va='bottom', transform=axs[4].transAxes, color='red', fontsize=10, weight='bold')
    # axs[4].text(0.98, 0.05, 'Flow diff RIFE', ha='right', va='bottom', transform=axs[4].transAxes, color='white', fontdict={'family': 'serif', 'size': 10})


    # Add separator lines between the rows
    for j in range(4):
        axs[j + 1].plot([0, 1], [1, 1], transform=axs[j + 1].transAxes, color='white', linewidth=line_thickness)

    # Remove bottom space and save the figure as an image file
    image_file = os.path.join(output_dir, f"flow_combined_frame_{i:03d}.png")
    plt.savefig(image_file, bbox_inches='tight', pad_inches=0)
    
    # Close the figure
    plt.close(fig)
    print(f"Frame {i} saved as {image_file}")

print("Image saving completed.")
