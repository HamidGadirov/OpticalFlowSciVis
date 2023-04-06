import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import cv2

def plot_distribution(data, dir_res, name="dist.png", save=False):
    fig = plt.figure(figsize=(8, 4))
    plt.hist(data.reshape(-1), bins=50)
    plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    if save:
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        res_path = os.path.join(dir_res, name)
        fig.savefig(res_path)

def load_files_from_folder(folder):
	filenames = []
	for filename in os.listdir(folder):
		# print(filename)
		if filename.endswith(".pkl"):
			if filename == "lbs2d_skip3_Cy1r1t1.pkl":
				# if filename != "lbs2d_skip3_ensemble20" + ".pkl" and filename != "lbs2d_skip3_ensemble21" + ".pkl":
				filenames.append(filename)

	# filenames.sort()
	print("len", len(filenames))

	return filenames
"""
filenames = load_files_from_folder(".")

data = []
dir_model = "./Distributions"

for filename in filenames:
	pkl_filename = filename
	with open(pkl_filename, 'rb') as pkl_file:
		data_x = pickle.load(pkl_file)
		print(data_x.shape)
		# data = np.concatenate((data, data_x[:3000]), axis=0)
		data.extend(data_x[:3000])
		data_x = np.array(data_x)
		print(pkl_filename)
		# print("Density range %f to %f" % (np.min(data_x[:, 0]), np.max(data_x[:, 0])))
		# print("Vel x range %f to %f" % (np.min(data_x[:, 1]), np.max(data_x[:, 1])))
		# print("Vel y range %f to %f" % (np.min(data_x[:, 2]), np.max(data_x[:, 2])))
		# plot_distribution(data_x[:, 1], dir_model, name=pkl_filename[:-4]+"_dist_flow_x.png", save=True)
		# plot_distribution(data_x[:, 2], dir_model, name=pkl_filename[:-4]+"_dist_flow_y.png", save=True)
		print(len(data))

# input("x")
data = np.array(data)
# data = np.float16(data)
# print("Converted to float16")
print(data.shape)
# input("x")

pkl_filename = "lbs2d_skip3_ensemble16m_f32" + ".pkl" 
pkl_file = open(pkl_filename, 'wb')
pickle.dump(data, pkl_file, protocol=4)
pkl_file.close
print("Pkl file created")
"""

data = []

pkl_filename = "lbs2d_skip3_1" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_1 = pickle.load(pkl_file)
pkl_file.close
print(data_1.shape)
pkl_filename = "lbs2d_skip3_2" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_2 = pickle.load(pkl_file)
pkl_file.close
print(data_2.shape)
pkl_filename = "lbs2d_skip3_3" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_3 = pickle.load(pkl_file)
pkl_file.close
print(data_3.shape)
pkl_filename = "lbs2d_skip3_Cy1r1t2" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_4 = pickle.load(pkl_file)
pkl_file.close
pkl_filename = "lbs2d_skip3_Cy1r1t3" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_5 = pickle.load(pkl_file)
pkl_file.close

"""
pkl_filename = "lbs2d_skip3_Cy1r1t1" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_1 = pickle.load(pkl_file)
pkl_file.close
print(data_1.shape)

pkl_filename = "lbs2d_skip3_Cy1r1t2" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_2 = pickle.load(pkl_file)
pkl_file.close
print(data_2.shape)

pkl_filename = "lbs2d_skip3_Cy1r1t3" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_3 = pickle.load(pkl_file)
pkl_file.close
print(data_3.shape)

pkl_filename = "lbs2d_skip3_Cx2y2r2t1" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_4 = pickle.load(pkl_file)
pkl_file.close
print(data_4.shape)

pkl_filename = "lbs2d_skip3_Cx3y3r3t1" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_5 = pickle.load(pkl_file)
pkl_file.close
print(data_5.shape)

pkl_filename = "lbs2d_skip3_Cy1r2t1" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_4 = pickle.load(pkl_file)
pkl_file.close
print(data_4.shape)

pkl_filename = "lbs2d_skip3_Cy1r2t2" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_5 = pickle.load(pkl_file)
pkl_file.close
print(data_5.shape)

pkl_filename = "lbs2d_skip3_Cy2r2t1" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_6 = pickle.load(pkl_file)
pkl_file.close
print(data_6.shape)

pkl_filename = "lbs2d_skip3_Cy4r2t1" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_7 = pickle.load(pkl_file)
pkl_file.close
print(data_7.shape)
"""

data = np.concatenate((data_1[:3000], data_2[:3000]), axis=0)
data = np.concatenate((data, data_3[:3000]), axis=0)
data = np.concatenate((data, data_4[:3000]), axis=0)
data = np.concatenate((data, data_5[:3000]), axis=0)
# data = np.concatenate((data, data_6[:3000]), axis=0)
# data = np.concatenate((data, data_7[:3000]), axis=0)

print(len(data))
pkl_filename = "lbs2d_skip3_combi_cyl_tau" + ".pkl" 
pkl_file = open(pkl_filename, 'wb')
pickle.dump(data, pkl_file, protocol=4)
pkl_file.close
print("Pkl file created")

# covert to float16
# and use amp
# and this causes explosion (?)