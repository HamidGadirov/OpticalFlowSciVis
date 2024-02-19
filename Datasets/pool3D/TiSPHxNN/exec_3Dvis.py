import numpy as np
import pyvista as pv

input_path = '../organized_output'

# Create a PyVista UniformGrid object
grid = pv.ImageData()

field = np.load('../organized_output/density_0.npz')['arr_0']

# Set the dimensions, spacing, and origin of the grid
grid.dimensions = np.array(field.shape) + 1
grid.spacing = (1, 1, 1)  # Assuming the spacing between your points is 1 in all directions
grid.origin = (0, 0, 0)  # Assuming the origin is at (0, 0, 0)

# Assign the density field to the grid
grid.cell_data["values"] = field.flatten(order="F")  # Flatten the array in Fortran order

# Create the volume plot
plotter = pv.Plotter()
# plotter.camera_position = 'zy'
# plotter.camera.elevation = 30
plotter.add_volume(grid, scalars="values")

# Display the plot
plotter.show()




# plotter_vel = pv.Plotter(off_screen=True)
# plotter_vel.open_gif("animation_vel.gif")

# plotter_density = pv.Plotter(off_screen=True)
# plotter_density.open_gif("animation_density.gif")

# for i in range(0, 1):
#     grid_density = pv.ImageData()

#     field_density = np.load(f'{input_path}/density_{i}.npz')['arr_0']

#     grid_density.dimensions = np.array(field_density.shape) + 1
#     grid_density.spacing = (1, 1, 1)  # Assuming the spacing between your points is 1 in all directions
#     grid_density.origin = (0, 0, 0)  # Assuming the origin is at (0, 0, 0)
#     grid_density.cell_data["values"] = field_density.flatten(order="F")  # Flatten the array in Fortran order
#     plotter_density.clear()
#     plotter_density.add_volume(grid_density, scalars="values")
#     outline = grid_density.outline_corners()
#     plotter_density.add_mesh(outline, color="black", show_edges=True)
#     plotter_density.camera_position = 'zy'
#     plotter_density.camera.elevation = 30
#     plotter_density.write_frame()  

#     grid_velocity = pv.ImageData()
#     field_vel = np.load(f'{input_path}/velocity_{i}.npz')['arr_0']
#     field_vel = np.sqrt(field_vel[...,0]**2+field_vel[...,1]**2+field_vel[...,2]**2)
#     grid_velocity.dimensions = np.array(field_vel.shape) + 1
#     grid_velocity.spacing = (1, 1, 1)  # Assuming the spacing between your points is 1 in all directions
#     grid_velocity.origin = (0, 0, 0)  # Assuming the origin is at (0, 0, 0)
#     grid_velocity.cell_data["values"] = field_vel.flatten(order="F")  # Flatten the array in Fortran order
#     plotter_vel.clear()
#     plotter_vel.add_volume(grid_velocity, scalars="values")
#     outline = grid_velocity.outline_corners()
#     plotter_vel.add_mesh(outline, color="black", show_edges=True)
#     plotter_vel.camera_position = 'zy'
#     plotter_vel.camera.elevation = 30
#     plotter_vel.write_frame()
    
# plotter_vel.close()
# plotter_density.close()

