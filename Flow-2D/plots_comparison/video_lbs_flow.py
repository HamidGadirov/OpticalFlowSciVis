import cv2
import os

# Set up the video writer
fps = 10  # Frames per second for the video
input_dir = "LBS_Flow_GT_Pred_Diff"
output_video = "flow_gt_video.mp4"

# Get list of image files
image_files = sorted([img for img in os.listdir(input_dir) if img.endswith(".png")])

# Ensure there are images to process
if not image_files:
    print("No images found in the specified directory.")
    exit()

# Read the first image to get the size
first_image_path = os.path.join(input_dir, image_files[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape
size = (width, height)

# Define the codec and create the video writer with a higher quality setting
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or cv2.VideoWriter_fourcc(*'H264') for higher quality
video = cv2.VideoWriter(output_video, fourcc, fps, size)

# Iterate through all image files and add them to the video
for img_file in image_files:
    img_path = os.path.join(input_dir, img_file)
    frame = cv2.imread(img_path)
    
    # Add frame to video
    video.write(frame)
    print(f"Added {img_file} to video.")

# Release the video writer
video.release()
print(f"Video creation completed: {output_video}")
