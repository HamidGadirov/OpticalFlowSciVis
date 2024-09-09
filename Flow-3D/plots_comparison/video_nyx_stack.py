import cv2

def stack_videos_vertically(video1_path, video2_path, output_path):
    # Open the two video files
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # Get video properties
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))

    # Open the video writer for the output
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height * 2))

    while True:
        # Read frames from both videos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Break the loop when any video ends
        if not ret1 or not ret2:
            break

        # Stack the frames vertically
        stacked_frame = cv2.vconcat([frame1, frame2])

        # Write the stacked frame to the output video
        out.write(stacked_frame)

    # Release everything when done
    cap1.release()
    cap2.release()
    out.release()

    print(f"Video {output_path} created successfully.")

if __name__ == "__main__":
    stack_videos_vertically('GT_interpol.mp4', 'GT_interpol_flow.mp4', 'stacked_output.mp4')
