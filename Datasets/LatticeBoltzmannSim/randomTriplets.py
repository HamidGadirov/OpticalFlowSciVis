import numpy as np

# Assuming you have an array of images called "data" with shape (60000, H, W)
num_members = 20
num_triplets = 60000

# Reshape data from (60000, H, W) to (20, 3000, H, W)
data_reshaped = data.reshape((num_members, -1, *data.shape[1:]))

# Create an empty array to store the triplets and corresponding t values
triplets = np.empty((num_triplets, 3, *data.shape[1:]))
t_values = np.empty(num_triplets)

# Randomly select triplets
for i in range(num_triplets):
    # Randomly select a member
    member_idx = np.random.randint(num_members)

    # Randomly select indices for the triplet
    k, t, l = np.random.choice(range(data_reshaped.shape[1]), size=3, replace=False)

    # Ensure k < t < l
    k, t, l = sorted([k, t, l])

    # Calculate t value
    t_value = (t - k) / (l - k)

    # Store the triplet and t value
    triplets[i] = np.concatenate((data_reshaped[member_idx, k], data_reshaped[member_idx, t], data_reshaped[member_idx, l]), axis=0)
    t_values[i] = t_value
