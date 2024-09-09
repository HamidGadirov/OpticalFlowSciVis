import os
import cv2
import numpy as np
import yt
import glob
from natsort import natsorted

"""
def render_density(datasets, labels, dir_res="Results"):
    bounds = {
        "jets3d": (0.1, 0.9),
        "nyx3d": (0.1, 0.7)
    }

    tf_settings = {
        "jets3d": [
            {"pos": 0.5, "width": 1e-4, "height": [0.05, 0.1, 1.0, 1e2]},  # Blue
            {"pos": 0.55, "width": 1e-4, "height": [1.0, 0.6, 0.1, 1e2]}  # Orange
        ],
        "nyx3d": [
            {"pos": 0.2, "width": 0.00025},
            {"pos": 0.3, "width": 0.00025},
            {"pos": 0.4, "width": 0.00025},
            {"pos": 0.5, "width": 0.00025}
        ]
    }

    for dataset, label in zip(datasets, labels):
        dataset_key = "jets3d" if "jets3d" in dataset else "nyx3d"
        bounds_value = bounds[dataset_key]
        tf = yt.ColorTransferFunction(np.log10(bounds_value), grey_opacity=True)

        for setting in tf_settings[dataset_key]:
            if dataset_key == "jets3d":
                tf.add_gaussian(np.log10(setting["pos"]), width=setting["width"], height=setting["height"])
            else:
                tf.sample_colormap(np.log10(setting["pos"]), w=setting["width"], alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")

        save_path = f"{dir_res}/Renderings_{label}"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        data_density = np.array(dataset)
        for idx in range(len(data_density)):
            volume = data_density[idx]
            ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
            sc = yt.create_scene(ds, field=("gas", "density"))
            source = sc[0]
            source.set_log(True)
            source.tfh.tf = tf
            source.tfh.bounds = bounds_value
            cam = sc.camera
            cam.resolution = 128
            cam.zoom(1.75 if "jets3d" in dataset else 1.0)
            cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
            im = sc.render()
            im.write_png(os.path.join(save_path, f'density_{700+idx}.png'), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
"""
def make_video(image_files_gt, image_files_rec, image_files_comparison):
    images_gt = [cv2.imread(img) for img in image_files_gt]
    images_rec = [cv2.imread(img) for img in image_files_rec]
    images_comparison = [cv2.imread(img) for img in image_files_comparison]

    fps = 5
    video = cv2.VideoWriter('GT_interpol_flow.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,
                            (images_gt[0].shape[1] * 3, images_gt[0].shape[0]))

    # Smaller font scale and thinner text
    font_scale = 0.6
    thickness = 1
    timestep = 303  # Initial timestep

    for idx, (img1, img2, img3) in enumerate(zip(images_gt, images_rec, images_comparison)):
        # img1 = cv2.putText(img1, 'GT flow', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        img1 = cv2.putText(img1, ' ', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        # img1 = cv2.putText(img1, f'T={timestep}', (20, 230), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        # img2 = cv2.putText(img2, 'FLINT flow', (160, 240), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        img2 = cv2.putText(img2, ' ', (160, 240), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # img3 = np.zeros_like(img1)  # Empty image for the third column
        # img3 = cv2.putText(np.zeros_like(img1), 'Nyx, 5x', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        img3 = cv2.putText(np.zeros_like(img1), ' ', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        concatenated_img = np.concatenate((img1, img2, img3), axis=1)
        concatenated_img = cv2.putText(concatenated_img, f'{idx}', (900, 500), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        video.write(concatenated_img)
        timestep += 5  # Increment timestep by 5 for the next frame

    video.release()
    print("Video created successfully.")


if __name__ == "__main__":

    # Load images from the directories
    data_path_gt = "Renderings_flow_GT"
    data_path_rec = "Renderings_flow"
    data_path_comparison = "Renderings_interpol_comparison"

    image_files_gt = natsorted(glob.glob(f'{data_path_gt}/flow_*.png'))
    image_files_rec = natsorted(glob.glob(f'{data_path_rec}/flow_*.png'))
    image_files_comparison = natsorted(glob.glob(f'{data_path_comparison}/density_*.png'))

    print(len(image_files_gt), len(image_files_rec), len(image_files_comparison))

    # render_density([gt_data, interpol_data, comparison_data], ['GT', 'interpol_ours', 'interpol_comparison'])

    # Create video from rendered images
    make_video(image_files_gt, image_files_rec, image_files_comparison)
