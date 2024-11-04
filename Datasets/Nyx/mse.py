import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
"""
# Function to load data from a pkl file
def load_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    data = data[0]
    print(f"Loaded data from {pkl_file}. Shape: {data.shape}")
    return data

# Function to compute 3D MSE between two volumes
def compute_mse(volume1, volume2):
    return np.mean((volume1 - volume2) ** 2)

# List of datasets
pkl_files = [
    'nyx3d_skip5_128_log10norm_OmM_012_OmB_00215_h_055_test_params.pkl', 
    'nyx3d_skip5_128_log10norm_OmM_012_OmB_00225_h_055_test_params.pkl', 
    'nyx3d_skip5_128_log10norm_OmM_012_OmB_00235_h_055_test_params.pkl', 

    'nyx3d_skip5_128_log10norm_OmM_01375_OmB_00215_h_055_test_params.pkl', 
    'nyx3d_skip5_128_log10norm_OmM_01375_OmB_00225_h_055_test_params.pkl', 
    'nyx3d_skip5_128_log10norm_OmM_01375_OmB_00235_h_055_test_params.pkl', 

    'nyx3d_skip5_128_log10norm_OmM_0155_OmB_00215_h_055_test_params.pkl', 
    'nyx3d_skip5_128_log10norm_OmM_0155_OmB_00225_h_055_test_params.pkl', 
    'nyx3d_skip5_128_log10norm_OmM_0155_OmB_00235_h_055_test_params.pkl'
]

# Load all datasets
datasets = [load_data(pkl_file) for pkl_file in pkl_files]

# Initialize similarity matrices
num_datasets = len(datasets)
similarity_matrix_density = np.zeros((num_datasets, num_datasets))
similarity_matrix_velocity = np.zeros((num_datasets, num_datasets))

# Scale factor for exponential similarity calculation
k = 200  # You can adjust this value

# Iterate over all pairs of datasets to compute MSE separately for density and velocity
for i in range(num_datasets):
    for j in range(i, num_datasets):
        print(f"\nComputing MSE between {pkl_files[i]} and {pkl_files[j]}:")

        # Compute MSE for density (index 0)
        density_mse = compute_mse(datasets[i][:, 0], datasets[j][:, 0])
        print(f"  Density MSE: {density_mse}")

        # Compute MSE for velocity components (indexes 1, 2, 3)
        velocity_mse_x = compute_mse(datasets[i][:, 1], datasets[j][:, 1])
        velocity_mse_y = compute_mse(datasets[i][:, 2], datasets[j][:, 2])
        velocity_mse_z = compute_mse(datasets[i][:, 3], datasets[j][:, 3])

        # Compute overall velocity MSE as the average of the three components
        velocity_mse = (velocity_mse_x + velocity_mse_y + velocity_mse_z) / 3
        print(f"  Velocity MSE: {velocity_mse} (X: {velocity_mse_x}, Y: {velocity_mse_y}, Z: {velocity_mse_z})")

        # Calculate similarity for density using exponential scaling
        similarity_density = np.exp(-density_mse * k)
        
        # Calculate similarity for velocity using exponential scaling
        similarity_velocity = np.exp(-velocity_mse * k)
        
        # Store in similarity matrices
        similarity_matrix_density[i, j] = similarity_density
        similarity_matrix_density[j, i] = similarity_density  # Symmetric
        
        similarity_matrix_velocity[i, j] = similarity_velocity
        similarity_matrix_velocity[j, i] = similarity_velocity  # Symmetric

# Extract only relevant parts of the filenames for axis labels
def clean_filename(filename):
    parts = filename.split('_')[4:8]  # Get only the parts like OmM_012, OmB_00215, h_055
    return '_'.join(parts)

# Function to plot the similarity matrix
def plot_similarity_matrix(similarity_matrix, pkl_files, field_name, save_dir):
    # Clean filenames for axis labels
    cleaned_filenames = [clean_filename(os.path.basename(f)) for f in pkl_files]

    # Set up the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the similarity matrix using a heatmap
    sns.heatmap(similarity_matrix, annot=True, cmap='viridis',
                xticklabels=cleaned_filenames, yticklabels=cleaned_filenames)
    
    # Adjust ticks and labels for clarity
    plt.xticks(rotation=45, ha="right")  # Rotate for x-axis labels
    plt.yticks(rotation=0)  # No rotation for y-axis labels
    
    plt.title(f'Similarity Matrix Based on {field_name} MSE (Exp Scaling)')
    plt.xlabel('Parameters')
    plt.ylabel('Parameters')

    # Save the plot
    plot_filename = os.path.join(save_dir, f"similarity_matrix_{field_name.lower()}_exp.png")
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()

    print(f"Similarity matrix plot for {field_name} saved at: {plot_filename}")

# Directory to save the plots
save_dir = './'

# Plot and save the similarity matrices for density and velocity
plot_similarity_matrix(similarity_matrix_density, pkl_files, 'Density', save_dir)
plot_similarity_matrix(similarity_matrix_velocity, pkl_files, 'Velocity', save_dir)
"""


# """
# Load the data from the pkl files
pkl_files = [
    'nyx3d_skip5_128_log10norm_OmM_012_OmB_00215_h_055_test_params.pkl', 
    'nyx3d_skip5_128_log10norm_OmM_012_OmB_00225_h_055_test_params.pkl', 
    'nyx3d_skip5_128_log10norm_OmM_012_OmB_00235_h_055_test_params.pkl', 

    'nyx3d_skip5_128_log10norm_OmM_01375_OmB_00215_h_055_test_params.pkl', 
    'nyx3d_skip5_128_log10norm_OmM_01375_OmB_00225_h_055_test_params.pkl', 
    'nyx3d_skip5_128_log10norm_OmM_01375_OmB_00235_h_055_test_params.pkl', 

    'nyx3d_skip5_128_log10norm_OmM_0155_OmB_00215_h_055_test_params.pkl', 
    'nyx3d_skip5_128_log10norm_OmM_0155_OmB_00225_h_055_test_params.pkl', 
    'nyx3d_skip5_128_log10norm_OmM_0155_OmB_00235_h_055_test_params.pkl'
]

# Function to load data from a pkl file
def load_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    data = data[0]
    print(f"Loaded data from {pkl_file}. Shape: {data.shape}")
    return data

# Function to compute 3D MSE between two volumes
def compute_mse(volume1, volume2):
    return np.mean((volume1 - volume2) ** 2)

# Load all datasets
datasets = [load_data(pkl_file) for pkl_file in pkl_files]

# Initialize similarity matrix
num_datasets = len(datasets)
similarity_matrix = np.zeros((num_datasets, num_datasets))

# Scale factor for exponential similarity calculation
k = 200  # You can adjust this value

# Iterate over all pairs of datasets to compute MSE for both density and velocity components
for i in range(num_datasets):
    for j in range(i, num_datasets):
        print(f"\nComputing MSE between {pkl_files[i]} and {pkl_files[j]}:")

        # Compute MSE for density (index 0)
        density_mse = compute_mse(datasets[i][:, 0], datasets[j][:, 0])
        print(f"  Density MSE: {density_mse}")

        # Compute MSE for velocity components (indexes 1, 2, 3)
        velocity_mse_x = compute_mse(datasets[i][:, 1], datasets[j][:, 1])
        velocity_mse_y = compute_mse(datasets[i][:, 2], datasets[j][:, 2])
        velocity_mse_z = compute_mse(datasets[i][:, 3], datasets[j][:, 3])

        # Compute overall velocity MSE as the average of the three components
        velocity_mse = (velocity_mse_x + velocity_mse_y + velocity_mse_z) / 3
        print(f"  Velocity MSE: {velocity_mse} (X: {velocity_mse_x}, Y: {velocity_mse_y}, Z: {velocity_mse_z})")

        # Compute average MSE (density and velocity)
        avg_mse = (density_mse + velocity_mse) / 2
        
        # Calculate similarity using exponential scaling
        similarity = np.exp(-avg_mse * k)
        
        # Store in similarity matrix
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity  # Symmetric

# Extract only relevant parts of the filenames for axis labels
def clean_filename(filename):
    parts = filename.split('_')[4:8]  # Get only the parts like OmM_012, OmB_00215, h_055
    return '_'.join(parts)

# Function to plot the similarity matrix
def plot_similarity_matrix(similarity_matrix, pkl_files, save_dir):
    # Clean filenames for axis labels
    cleaned_filenames = [clean_filename(os.path.basename(f)) for f in pkl_files]

    # Set up the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the similarity matrix using a heatmap
    sns.heatmap(similarity_matrix, annot=True, cmap='viridis',
                xticklabels=cleaned_filenames, yticklabels=cleaned_filenames)
    
    # Adjust ticks and labels for clarity
    plt.xticks(rotation=45, ha="right")  # Rotate for x-axis labels
    plt.yticks(rotation=0)  # No rotation for y-axis labels
    
    plt.title('Similarity Matrix Based on MSE (Exponential Scaling)')
    plt.xlabel('Datasets')
    plt.ylabel('Datasets')

    # Save the plot
    plot_filename = os.path.join(save_dir, "mse_similarity_matrix_exp.png")
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.close()

    print(f"Similarity matrix plot saved at: {plot_filename}")

# Directory to save the plot
save_dir = './'

# Plot and save the similarity matrix
plot_similarity_matrix(similarity_matrix, pkl_files, save_dir)
# """


"""
import os
import pickle
import numpy as np

# Load the data from the pkl files
pkl_files = [
    'nyx3d_skip5_128_log10norm_OmM_012_OmB_00215_h_055_test_params.pkl', # +
    'nyx3d_skip5_128_log10norm_OmM_012_OmB_00225_h_055_test_params.pkl', # +
    'nyx3d_skip5_128_log10norm_OmM_012_OmB_00235_h_055_test_params.pkl', # +

    'nyx3d_skip5_128_log10norm_OmM_01375_OmB_00215_h_055_test_params.pkl', # +
    'nyx3d_skip5_128_log10norm_OmM_01375_OmB_00225_h_055_test_params.pkl', # +
    'nyx3d_skip5_128_log10norm_OmM_01375_OmB_00235_h_055_test_params.pkl', # +

    'nyx3d_skip5_128_log10norm_OmM_0155_OmB_00215_h_055_test_params.pkl', # +
    'nyx3d_skip5_128_log10norm_OmM_0155_OmB_00225_h_055_test_params.pkl', # +
    'nyx3d_skip5_128_log10norm_OmM_0155_OmB_00235_h_055_test_params.pkl' # +
]

# Function to load data from a pkl file
def load_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    # Extract the first element from the tuple (which is the data without params)
    data = data[0]
    print(f"Loaded data from {pkl_file}. Shape: {data.shape}")
    return data

# Function to compute 3D MSE between two volumes
def compute_mse(volume1, volume2):
    return np.mean((volume1 - volume2) ** 2)

# Load all datasets
datasets = [load_data(pkl_file) for pkl_file in pkl_files]

# Iterate over all pairs of datasets to compute MSE for both density and velocity components
for i in range(len(datasets)):
    for j in range(i + 1, len(datasets)):
        print(f"\nComputing MSE between {pkl_files[i]} and {pkl_files[j]}:")

        # Compute MSE for density (index 0)
        density_mse = compute_mse(datasets[i][:, 0], datasets[j][:, 0])
        print(f"  Density MSE: {density_mse}")

        # Compute MSE for velocity components (indexes 1, 2, 3)
        velocity_mse_x = compute_mse(datasets[i][:, 1], datasets[j][:, 1])
        velocity_mse_y = compute_mse(datasets[i][:, 2], datasets[j][:, 2])
        velocity_mse_z = compute_mse(datasets[i][:, 3], datasets[j][:, 3])

        # Compute overall velocity MSE as the average of the three components
        velocity_mse = (velocity_mse_x + velocity_mse_y + velocity_mse_z) / 3
        print(f"  Velocity MSE: {velocity_mse} (X: {velocity_mse_x}, Y: {velocity_mse_y}, Z: {velocity_mse_z})")

"""

"""
import os
import pickle
import numpy as np

# Load the data from the pkl files
pkl_files = [
    'nyx3d_skip5_128_log10norm_OmM_012_OmB_00215_h_055_test_params.pkl',
    'nyx3d_skip5_128_log10norm_OmM_012_OmB_00235_h_055_test_params.pkl',
    'nyx3d_skip5_128_log10norm_OmM_0155_OmB_00215_h_055_test_params.pkl',
    'nyx3d_skip5_128_log10norm_OmM_0155_OmB_00235_h_055_test_params.pkl'
]

# Function to load data from a pkl file
def load_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    # Extract the first element from the tuple (which is the data without params)
    data = data[0]
    print(f"Loaded data from {pkl_file}. Shape: {data.shape}")
    return data

# Function to compute 3D MSE between two volumes
def compute_mse(volume1, volume2):
    return np.mean((volume1 - volume2) ** 2)

# Load all datasets
datasets = [load_data(pkl_file) for pkl_file in pkl_files]


# Iterate over all pairs of datasets to compute MSE per timestep for both density and velocity components
for i in range(len(datasets)):
    for j in range(i + 1, len(datasets)):
        print(f"\nComputing MSE between {pkl_files[i]} and {pkl_files[j]}:")

        # Iterate over each timestep (160 in total)
        for t in range(datasets[i].shape[0]):
            print(f"Timestep {t}:")

            # Compute MSE for density (index 0)
            density_mse = compute_mse(datasets[i][t, 0], datasets[j][t, 0])
            print(f"  Density MSE: {density_mse}")

            # Compute MSE for velocity components (indexes 1, 2, 3)
            velocity_mse_x = compute_mse(datasets[i][t, 1], datasets[j][t, 1])
            velocity_mse_y = compute_mse(datasets[i][t, 2], datasets[j][t, 2])
            velocity_mse_z = compute_mse(datasets[i][t, 3], datasets[j][t, 3])

            # Compute overall velocity MSE as the average of the three components
            velocity_mse = (velocity_mse_x + velocity_mse_y + velocity_mse_z) / 3
            print(f"  Velocity MSE: {velocity_mse} (X: {velocity_mse_x}, Y: {velocity_mse_y}, Z: {velocity_mse_z})")
"""
