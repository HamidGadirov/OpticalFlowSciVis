import cv2
import os

# Path to the folder containing the images
path = 'output0.05_part'

# Get the list of files in the folder
files = sorted(os.listdir(path))

# Set the frame rate (in frames per second)
fps = 30.0

# Set the video codec
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# Get the first image to determine the size of the video
img = cv2.imread(os.path.join(path, files[0]))
height, width, channels = img.shape

# Create the video writer object
video = cv2.VideoWriter(path+'.avi', fourcc, fps, (1920, 1080))

# Loop through all the images in the folder and add them to the video
i = 0
for filename in files:
    print(str(i)+'.png')
    img = cv2.imread(os.path.join(path, str(i)+'.png'))
    i += 1
    video.write(img)

# Release the video writer object
video.release()