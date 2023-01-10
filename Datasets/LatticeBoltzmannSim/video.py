import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import cv2

def load_images_from_folder(folder):
	images = []
	filenames = []
	for filename in os.listdir(folder):
		# print(filename)
		if filename.endswith(".png"):
			filenames.append(filename)

	filenames.sort()
	print("len", len(filenames))

	for filename in filenames:
		print(filename)
		img = cv2.imread(os.path.join(folder, filename))
		if img is not None:
			images.append(img)
			# print("x")
	return images

images = load_images_from_folder("video")
print(len(images))
images = np.array(images)
print(images.shape)
# input("X")

_, height, width, layers = images.shape
size = (width,height)

video_name = "LBS_Vorticity_Density_Velocity" + "_10fps.mp4"
fps = 10
# sizeY = 480
# sizeX = 640
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
for i in range(len(images)):
	# out.write(images[i].astype('uint8'))
	out.write(images[i])
	# print(i)
out.release()
# input("X")
# input("X")