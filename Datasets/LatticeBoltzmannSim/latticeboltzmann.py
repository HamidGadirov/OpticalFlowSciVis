import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import cv2

"""
Create Your Own Lattice Boltzmann Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate flow past cylinder
for an isothermal fluid
"""

"""
added fields and created dataset
Hamid Gadirov (2022) Univeristy of Groningen
"""
# def generate_video(img):
#     for i in range(len(img)):
#         plt.imshow(img[i], cmap='bwr') # cm.Greys_r)
#         plt.savefig("video" + "/file%02d.png" % i)

#     os.chdir("your_folder")
#     subprocess.call([
#         'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
#         'video_name.mp4'
#     ])
#     for file_name in glob.glob("*.png"):
#         os.remove(file_name)

def main():
	""" Lattice Boltzmann Simulation """
	
	# Simulation parameters
	Nx                     = 400    # resolution x-dir
	Ny                     = 100    # resolution y-dir
	rho0                   = 100    # average density
	tau                    = 0.6    # collision timescale
	Nt                     = 30000   # number of timesteps # 4000
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Lattice speeds / weights
	NL = 9
	idxs = np.arange(NL)
	cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
	cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
	weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1
	
	# Initial Conditions
	F = np.ones((Ny,Nx,NL)) #* rho0 / NL
	np.random.seed(42)
	F += 0.01*np.random.randn(Ny,Nx,NL)
	X, Y = np.meshgrid(range(Nx), range(Ny))
	F[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/Nx*4))
	rho = np.sum(F,2)
	for i in idxs:
		F[:,:,i] *= rho0 / rho
	
	# Cylinder boundary
	X, Y = np.meshgrid(range(Nx), range(Ny))
	cylinder = (X - Nx/5)**2 + (Y - Ny/2.5)**2 < (Ny/4)**2
	
	# Prep figure
	fig = plt.figure(figsize=(4,2), dpi=80)

	density = []
	vel_x = []
	vel_y = []
	magnitude = []
	vort = []
	# Simulation Main Loop
	for it in range(Nt):
		print(it)
		
		# Drift
		for i, cx, cy in zip(idxs, cxs, cys):
			F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
			F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
		
		# Set reflective boundaries
		bndryF = F[cylinder,:]
		bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]
	
		# Calculate fluid variables
		rho = np.sum(F,2)
		ux  = np.sum(F*cxs,2) / rho
		uy  = np.sum(F*cys,2) / rho
		# print(ux.shape)
		# print("ux uy at 30 100", ux[30][100], uy[30][100])
		# print("ux uy at 50 200", ux[50][200], uy[50][200])
		# print("ux uy at 70 300", ux[70][300], uy[70][300])
		# print("ux uy mean grid", np.mean(ux), np.mean(uy))
		# print("rho mean grid", np.mean(rho), np.mean(rho))
		# print("rho at 30 100", rho[30][100])
		# print("rho at 50 200", rho[50][200])
		# print("rho at 70 300", rho[70][300])
		
		
		# Apply Collision
		Feq = np.zeros(F.shape)
		for i, cx, cy, w in zip(idxs, cxs, cys, weights):
			Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )
		
		F += -(1.0/tau) * (F - Feq)
		
		# Apply boundary 
		F[cylinder,:] = bndryF

		
		# plot in real time - color 1/2 particles blue, other half red
		if (plotRealTime and (it % 10) == 0) or (it == Nt-1):
			plt.cla()
			# ux[cylinder] = 0
			# uy[cylinder] = 0
			# vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
			# vorticity[cylinder] = np.nan
			# vorticity = np.ma.array(vorticity, mask=cylinder)
			# plt.imshow(vorticity, cmap='bwr')
			# plt.imshow(~cylinder, cmap='gray', alpha=0.3)
			# plt.clim(-.1, .1)
			# ax = plt.gca()
			# ax.invert_yaxis()
			# ax.get_xaxis().set_visible(False)
			# ax.get_yaxis().set_visible(False)	
			# ax.set_aspect('equal')	
			# plt.pause(0.001)

			# plt.cla()
			# rho[cylinder] = 0
			# rho = np.ma.array(rho, mask=cylinder)
			# plt.imshow(rho, cmap='bwr')
			# plt.imshow(~cylinder, cmap='gray', alpha=0.3)
			# plt.clim(-.1, .1)
			# ax = plt.gca()
			# ax.invert_yaxis()
			# ax.get_xaxis().set_visible(False)
			# ax.get_yaxis().set_visible(False)	
			# ax.set_aspect('equal')	
			# plt.pause(0.001)

			norm = np.sqrt(ux * ux + uy * uy) # direction???

			# plt.cla()
			# norm = np.ma.array(norm, mask=cylinder)
			# plt.imshow(norm, cmap='bwr')
			# plt.imshow(~cylinder, cmap='gray', alpha=0.3)
			# plt.clim(-.1, .1)
			# ax = plt.gca()
			# ax.invert_yaxis()
			# ax.get_xaxis().set_visible(False)
			# ax.get_yaxis().set_visible(False)	
			# ax.set_aspect('equal')	
			# plt.pause(0.001)

			# print("norm at 30 100", norm[30][100])
			# print("norm at 50 200", norm[50][200])
			# print("norm at 70 300", norm[70][300])

			fig, axs = plt.subplots(3)
			fig.suptitle('Vorticity, density, velocity')

			# plt.cla()
			ux[cylinder] = 0
			uy[cylinder] = 0
			vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
			vorticity[cylinder] = np.nan
			vorticity = np.ma.array(vorticity, mask=cylinder)
			# axs[0].imshow(vorticity, cmap='bwr')
			# axs[0].imshow(~cylinder, cmap='gray', alpha=0.3)
			# plt.clim(-.1, .1)
			# axs[0].clim(-.1, .1)
			# ax = plt.gca()
			# ax.invert_yaxis()
			# ax.get_xaxis().set_visible(False)
			# ax.get_yaxis().set_visible(False)	
			# ax.set_aspect('equal')	
			# plt.pause(0.001)

			# plt.cla()
			rho = np.ma.array(rho, mask=cylinder)
			# axs[1].imshow(rho, cmap='bwr')
			# axs[1].imshow(~cylinder, cmap='gray', alpha=0.3)

			norm = np.ma.array(norm, mask=cylinder)
			# axs[2].imshow(norm, cmap='bwr')
			# axs[2].imshow(~cylinder, cmap='gray', alpha=0.3)
			# plt.clim(-.1, .1)
			# ax = plt.gca()
			# plt.pause(0.01)
			plt.close()

			# # ret, frame = cap.read()
			# final_frame = cv2.vconcat((vorticity, rho))
			# final_frame = cv2.vconcat((final_frame, norm))
			# # print(final_frame.shape)
			# # cv2.imshow('frame',final_frame)
			# out.write(final_frame.astype('uint8'))
			# 	# out.write(np.invert(data_for_interpol[i].astype('uint8'))) # try invert for droplet2d

			if it == 3000:
				# fig.savefig("Vorticity, density, velocity t=3000")
				# out.release()
				# print("Created:", video_name)
				# input("x")
				break

			# # vel_x = np.ma.array(vel_x, mask=cylinder)
			# # axs[1].imshow(vel_x, cmap='bwr')
			# # axs[1].imshow(~cylinder, cmap='gray', alpha=0.3)
		
			# # add density and velolicites to list
			# # each 10th timestep
			# density.append(rho)
			# vel_x.append(ux)
			# vel_y.append(uy)
		
			# add density and velolicites to list
			# each 10th timestep
			density.append(rho)
			vel_x.append(ux)
			vel_y.append(uy)
			magnitude.append(norm)
			vort.append(vorticity)

	vorticity = np.array(vort)
	density = np.array(density)
	vel_x = np.array(vel_x)
	vel_y = np.array(vel_y)
	magnitude = np.array(magnitude)
	print("vorticity:", vorticity.shape)
	print("density:", density.shape)
	print("vel_x:", vel_x.shape)
	print("magnitude:", magnitude.shape)
	# input("X")

	video = np.concatenate((vorticity, density), axis=1)
	video = np.concatenate((video, magnitude), axis=1)
	print("video data:", video.shape)

	# video_name = "LBS_Vorticity_Density_Velocity" + "_10fps.mp4"
	# fps = 10
	# sizeY = 400
	# sizeX = 300
	# out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (sizeY, sizeX), False)
	# for i in range(len(video)):
	# 	out.write(video[i].astype('uint8'))
    #     # out.write(np.invert(data_for_interpol[i].astype('uint8'))) # try invert for droplet2d
	# out.release()
	# input("X")
	# input("X")

	fps = 10
	sizeY = 400
	sizeX = 100

	video_name = "LBS_Vorticity" + "_10fps.mp4"
	out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (sizeY, sizeX), False)
	for i in range(len(vorticity)):
		out.write(vorticity[i].astype('uint8'))
        # out.write(np.invert(data_for_interpol[i].astype('uint8'))) # try invert for droplet2d
	out.release()
	input("Vorticity")

	video_name = "LBS_Density" + "_10fps.mp4"
	out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (sizeY, sizeX), False)
	for i in range(len(density)):
		out.write(density[i].astype('uint8'))
        # out.write(np.invert(data_for_interpol[i].astype('uint8'))) # try invert for droplet2d
	out.release()
	input("Density")

	video_name = "LBS_Velocity" + "_10fps.mp4"
	out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (sizeY, sizeX), False)
	for i in range(len(magnitude)):
		out.write(magnitude[i].astype('uint8'))
        # out.write(np.invert(data_for_interpol[i].astype('uint8'))) # try invert for droplet2d
	out.release()
	input("Velocity")

	# for i in range(len(density)):
	# 	out.write(density[i].astype('uint8'))
    #     # out.write(np.invert(data_for_interpol[i].astype('uint8'))) # try invert for droplet2d
	# out.release()
	# input("X")
	
	# # Save figure
	# plt.savefig('latticeboltzmann.png',dpi=240)
	# plt.show()

	# save data to pkl
	pkl_filename = "lbs2d_skip" + ".pkl" 

	# data = np.zeros((Nt, Ny, Nx), dtype=np.float32)
	data = density
	print("data:", data.shape)
	data = data[:, np.newaxis, ...]
	velocities_x = vel_x[:, np.newaxis, ...]
	velocities_y = vel_y[:, np.newaxis, ...]
	data = np.hstack((data, velocities_x))
	data = np.hstack((data, velocities_y))
	print(data.shape)
	# input("x")

	data = np.float32(data)

	pkl_file = open(pkl_filename, 'wb')
	pickle.dump(data, pkl_file, protocol=4)
	pkl_file.close
	print("Pkl file created")

	pkl_file = open(pkl_filename, 'rb')
	data = []
	data = pickle.load(pkl_file)
	pkl_file.close
	print(data.shape)
	    
	return 0


if __name__== "__main__":
  main()

