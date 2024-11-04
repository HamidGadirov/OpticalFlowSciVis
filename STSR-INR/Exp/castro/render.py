import os
import yt
import pickle
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from tqdm import tqdm

class VolumeRenderer:
    def __init__(self, eval_dirPath, gt_filePath):
        self.eval_dirPath = eval_dirPath
        self.gt_filePath = gt_filePath
        self.PSNR = []
        self.MSE = []

    def load_eval_data(self, eval_file):
        """Load evaluation data from a .raw file."""
        with open(eval_file, 'rb') as f:
            eval_data = np.fromfile(f, dtype=np.float32)  # Assuming float32 is the correct data type
        eval_data = eval_data.reshape((128, 128, 128), order='F')  # Ensure Fortran order
        return eval_data

    def load_gt_data(self):
        """Load ground truth data from a .pkl file."""
        with open(self.gt_filePath, 'rb') as f:
            data = pickle.load(f)
        data = data[0] # without params
        data = data[:160]
        v = data[:, 0]  # Assuming the first variable is the density field
        v = v.reshape(-1, 128, 128, 128)  # Reshape to (num_segments, 128, 128, 128)
        return v

    def normalize_data(self, data):
        """Normalize the data to [0, 1]."""
        data_min = data.min()
        data_max = data.max()
        return (data - data_min) / (data_max - data_min)

    def render_volume_yt(self, volume, save_path, file_name):
        """Render the 3D volume using yt and save the image."""
        ds_numpy = yt.load_uniform_grid(dict(density=volume), volume.shape)
        sc = yt.create_scene(ds_numpy, field=("gas", "density"))
        source = sc[0]

        source.set_log(True)
        bounds = (0.1, 0.7)
        tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf.sample_colormap(np.log10(0.2), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf.sample_colormap(np.log10(0.3), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf.sample_colormap(np.log10(0.4), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf.sample_colormap(np.log10(0.5), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")

        source.tfh.tf = tf
        source.tfh.bounds = bounds

        cam = sc.camera
        cam.resolution = 256
        cam.zoom(1.0)
        im = sc.render()
        im.write_png(os.path.join(save_path, file_name), background=[0.0, 0.0, 0.0, 1.0], sigma_clip=4.0)  # 2.0
        print("Rendered:", file_name)

    def calculate_psnr(self, gt_data, eval_data):
        """Calculate PSNR between GT and evaluation data."""
        return peak_signal_noise_ratio(gt_data, eval_data, data_range=gt_data.max() - gt_data.min())

    def calculate_mse(self, gt_data, eval_data):
        """Calculate MSE between GT and evaluation data."""
        return mean_squared_error(gt_data, eval_data)

    def getPSNR(self):
        """Get PSNR from the GT and eval data."""
        # Load GT data
        GT_data = self.load_gt_data()
        print(f"GT Data Loaded: shape={GT_data.shape}, min={GT_data.min()}, max={GT_data.max()}, mean={GT_data.mean()}")

        # Fetch eval file paths
        filePaths_eval = [os.path.join(self.eval_dirPath, f) for f in os.listdir(self.eval_dirPath) if f.endswith('.raw')]
        filePaths_eval.sort()  # Ensure the files are processed in order

        # Check that GT_data can be divided into segments corresponding to eval files
        if len(GT_data) % len(filePaths_eval) != 0:
            raise ValueError("GT data length is not divisible by the number of eval files")

        segment_length = len(GT_data) // len(filePaths_eval)

        # Calculate PSNR and MSE for each file pair
        for index in tqdm(range(len(filePaths_eval)), desc="Calculating Metrics"):
            eval_data = self.load_eval_data(filePaths_eval[index])
            print(f"Eval Data Loaded (Index {index}): shape={eval_data.shape}, min={eval_data.min()}, max={eval_data.max()}, mean={eval_data.mean()}")

            gt_segment = GT_data[index * segment_length : (index + 1) * segment_length]
            
            # Normalize the data before calculating PSNR and MSE
            gt_segment_norm = self.normalize_data(gt_segment[0])
            eval_data_norm = self.normalize_data(eval_data)
            
            print(f"Normalized GT Segment (Index {index}): min={gt_segment_norm.min()}, max={gt_segment_norm.max()}, mean={gt_segment_norm.mean()}")
            print(f"Normalized Eval Data (Index {index}): min={eval_data_norm.min()}, max={eval_data_norm.max()}, mean={eval_data_norm.mean()}")

            # Render the volumes
            self.render_volume_yt(gt_segment_norm, 'renders', f'gt_segment_{index}.png')
            self.render_volume_yt(eval_data_norm, 'renders', f'eval_data_{index}.png')
            
            psnr_value = self.calculate_psnr(gt_segment_norm, eval_data_norm)  # Use only the density data
            mse_value = self.calculate_mse(gt_segment_norm, eval_data_norm)
            print("mse_value", mse_value)
            self.PSNR.append(psnr_value)
            self.MSE.append(mse_value)

        MeanPSNR = np.array(self.PSNR).mean()
        MeanMSE = np.array(self.MSE).mean()
        return MeanPSNR, self.PSNR, MeanMSE, self.MSE

# Usage
eval_dirPath = 'vcastro/Results/castro_Scalar'
# gt_filePath = '../../../Datasets/castro/castro_ensemble_test/Nyx/Exec/LyA/nyx3d_skip5_128_log10norm_OmM_01375_OmB_00225_h_055_STSRINR_train.pkl'
gt_filePath = '../../../Datasets/Castro/Exec/science/wdmerger/castro3d_128_log10norm_massP_090_massS_085_test_params.pkl'
renderer = VolumeRenderer(eval_dirPath, gt_filePath)
mean_psnr, psnr_list, mean_mse, mse_list = renderer.getPSNR()
print(f"Mean PSNR: {mean_psnr}")
print(f"Mean MSE: {mean_mse}")