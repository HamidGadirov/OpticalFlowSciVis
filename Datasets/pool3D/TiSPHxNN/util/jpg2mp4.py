import cv2
import os

# Output MP4 file path
output_path = './output'
input_path = output_path

# Frame rate (frames per second) of the output video
start_index = 0
end_index = 186
frame_rate = 24
stride = 1


def process(file_name):
    # Get the list of image files sorted numerically
    image_files = []
    for i in range(start_index,end_index,stride):
        image_files.append(f'./{input_path}/{file_name}_{i}.png')

    # Get the first image to determine the frame size
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape

    print(height, width, layers)

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
    out = cv2.VideoWriter(f'{output_path}/{file_name}.mp4', fourcc, frame_rate, (width, height))

    # Iterate through the image files and write frames to the video
    for image_file in image_files:
        image = cv2.imread(image_file)
        out.write(image)

    cv2.destroyAllWindows()
    # Release the video writer
    out.release()

# process('sci_input_velocity')
process('sci_output_velocity')
# process('sci_input_vorticity')
process('sci_output_vorticity')
# process('input_vorticity_hist')
process('output_vorticity_hist')
# process('strainRate_compression')
# process('strainRate_rotation')
# process('strainRate_shear')
# process('vel_hsv')