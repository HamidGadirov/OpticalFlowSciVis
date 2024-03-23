import pickle
import numpy as np

print("loading...")
# Load data from the pickle file
with open('pool3d_128_triplets_W12_500_train.pkl', 'rb') as f:
    data = pickle.load(f)

print("loaded from pickle")

# Convert to numpy array
data_np = np.array(data)
print(data_np.dtype)

# Save as .npy file
np.save('pool3d_128_triplets_W12_500_train.npy', data_np.astype(np.float32))