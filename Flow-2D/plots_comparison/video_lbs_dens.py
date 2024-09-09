import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

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

# Set up video writers
fps = 10
size = (400, 100 * 5)  # Dimensions for lbs2d
video1 = cv2.VideoWriter('flow_data_video_new.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

# Density Data Visualization
cmap = 'turbo'
vmin = 95
vmax = original_data.max()

print(len(original_data), len(interpol_data_LREST), len(diffs_LREST), len(interpol_data_RIFE), len(diffs_RIFE))

# Determine the length of the shortest data array
min_length = min(len(original_data), len(interpol_data_LREST), len(diffs_LREST),
                 len(interpol_data_RIFE), len(diffs_RIFE))

# Define labels for each row
labels = ["Dens GT", "Dens FLINT", "Dens diff x100", "Dens RIFE", "Dens diff x100"]

# Font settings for the labels
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
thickness = 1
font_color = (255, 255, 255)
padding = 5  # Padding between the label and the edge of the frame

# Adjusted label positions
label_positions = [(size[0] - padding, (i + 1) * (size[1] // len(labels)) - padding) for i in range(len(labels))]

# Thickness of the separating lines
line_thickness = 2

# Timestep settings
initial_timestep = 4605
timestep_increment = 1

for i in range(min_length):
    # Create frames for each field
    frame_original = generate_frame(original_data[i], cmap, vmin, vmax)
    frame_interpol_LREST = generate_frame(interpol_data_LREST[i], cmap, vmin, vmax)
    frame_diffs_LREST = generate_frame(diffs_LREST[i] * 100, cmap, vmin=original_data.min(), vmax=vmax)  # magnified by 100
    frame_interpol_RIFE = generate_frame(interpol_data_RIFE[i], cmap, vmin, vmax)
    frame_diffs_RIFE = generate_frame(diffs_RIFE[i] * 100, cmap, vmin=original_data.min(), vmax=vmax)  # magnified by 100

    # Stack the frames vertically
    combined_frame = np.vstack((frame_original, frame_interpol_LREST, frame_diffs_LREST, frame_interpol_RIFE, frame_diffs_RIFE))

    # Resize the combined frame to fit the video size
    combined_frame = cv2.resize(combined_frame, size)

    # Add labels to each row
    # for j, (label, position) in enumerate(zip(labels, label_positions)):
    #     text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    #     text_x = position[0] - text_size[0] - padding  # Adjust to right-align text
    #     text_y = position[1] - padding  # Adjust to position the text within the row
    #     cv2.putText(combined_frame, label, (text_x, text_y), font, font_scale, font_color, thickness)

    # Add timestep information to the top right corner
    timestep = initial_timestep + i * timestep_increment
    timestep_text = f'T={timestep}'
    text_size = cv2.getTextSize(timestep_text, font, font_scale, thickness)[0]
    text_x = combined_frame.shape[1] - text_size[0] - padding  # Right-align the text
    text_y = text_size[1] + padding  # Align text near the top
    cv2.putText(combined_frame, timestep_text, (text_x-8, text_y+5), font, 0.6, (0, 0, 0), thickness=2)

    # Draw black lines to separate each row
    row_height = size[1] // len(labels)
    for j in range(1, len(labels)):
        y = j * row_height
        cv2.line(combined_frame, (0, y), (size[0], y), (0, 0, 0), thickness=line_thickness)

    # Write the frame to the video
    video1.write(combined_frame)

video1.release()
