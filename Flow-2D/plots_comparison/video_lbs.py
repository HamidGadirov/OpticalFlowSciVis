import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import pyimof

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

# Load density data
original_data = load_data('original_data.pkl')
interpol_data_LREST = load_data('interpol_data_LREST.pkl')
diffs_LREST = load_data('diffs_LREST.pkl')
interpol_data_RIFE = load_data('interpol_data_RIFE.pkl')
diffs_RIFE = load_data('diffs_RIFE.pkl')

# Load flow data
flow_gt = load_data('flow_gt.pkl')
flow_pred_LREST = load_data('flow_pred_LREST.pkl')
flow_diff_LREST = load_data('flow_diff_LREST.pkl')
flow_pred_RIFE = load_data('flow_pred_RIFE.pkl')
flow_diff_RIFE = load_data('flow_diff_RIFE.pkl')

# Set up video writers
fps = 30
size = (400, 100*5)  # Dimensions for lbs2d
video1 = cv2.VideoWriter('density_data_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
video2 = cv2.VideoWriter('flow_data_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

# Density Data Visualization
cmap = 'turbo'
vmin = 95
vmax = original_data.max()

print(len(original_data), len(interpol_data_LREST), len(diffs_LREST), len(interpol_data_RIFE), len(diffs_RIFE))

# Determine the length of the shortest data array
min_length = min(len(original_data), len(interpol_data_LREST), len(diffs_LREST),
                 len(interpol_data_RIFE), len(diffs_RIFE))

for i in range(min_length):
    # Create frames for each field
    frame_original = generate_frame(original_data[i], cmap, vmin, vmax)
    frame_interpol_LREST = generate_frame(interpol_data_LREST[i], cmap, vmin, vmax)
    frame_diffs_LREST = generate_frame(diffs_LREST[i], cmap, vmin, vmax)
    frame_interpol_RIFE = generate_frame(interpol_data_RIFE[i], cmap, vmin, vmax)
    frame_diffs_RIFE = generate_frame(diffs_RIFE[i], cmap, vmin, vmax)

    # Stack the frames vertically and resize to fit the video size
    combined_frame = np.vstack((frame_original, frame_interpol_LREST, frame_diffs_LREST, frame_interpol_RIFE, frame_diffs_RIFE))
    combined_frame = cv2.resize(combined_frame, size)
    
    video1.write(combined_frame)

video1.release()

# Flow Data Visualization
quiver_steps = 4
max_flow_gt = np.max([np.max(np.sqrt(u**2 + v**2)) for field in [flow_gt, flow_pred_LREST, flow_diff_LREST, flow_pred_RIFE, flow_diff_RIFE] for u, v in zip(field[:, 0, ...], field[:, 1, ...])])
scale = None

# Desired video frame size and DPI
frame_width, frame_height = 400, 100  # Width and height in pixels
dpi = 100  # Adjust DPI if necessary

# Calculate the total figure height for the subplots
total_fig_height = (frame_height * 5) / dpi  # Total height in inches for 5 subplots

for i in range(min_length):
    # Create a figure with the correct size
    fig, axs = plt.subplots(5, 1, figsize=(frame_width / dpi, total_fig_height), dpi=dpi)

    # Create subplots for each field
    fields = [flow_gt, flow_pred_LREST, flow_diff_LREST, flow_pred_RIFE, flow_diff_RIFE]
    for j, field in enumerate(fields):
        u = field[i, 0, ...]
        v = field[i, 1, ...]
        norm = np.sqrt(u ** 2 + v ** 2)
        img = np.zeros(u.shape)
        pyimof.display.quiver(u, v, c=norm, bg=img, ax=axs[j], cmap='turbo', clim=(0, max_flow_gt), scale=None, bg_cmap='gray', step=quiver_steps)
        axs[j].axis('off')

    # Convert the Matplotlib figure to an OpenCV frame
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (frame_width, frame_height * 5))  # Resize to match the video size

    video2.write(frame)
    plt.close(fig)

video2.release()





