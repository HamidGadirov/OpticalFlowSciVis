import cv2

def stack_videos_horizontally(video_path_1, video_path_2, output_path):
    # Open the two videos
    cap1 = cv2.VideoCapture(video_path_1)
    cap2 = cv2.VideoCapture(video_path_2)

    # Get properties from both videos
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine the new width to make both frames the same width
    if width1 > width2:
        new_width2 = width1
        new_width1 = width1
    else:
        new_width2 = width2
        new_width1 = width2

    # Use the maximum height and ensure both frames have the same width
    output_width = new_width1 + new_width2
    output_height = max(height1, height2)
    
    # Define the codec and create VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    # Read and process each frame
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Resize frames with high-quality interpolation to ensure both have the same height
        if height1 != height2:
            frame1 = cv2.resize(frame1, (new_width1, output_height), interpolation=cv2.INTER_LINEAR)
            frame2 = cv2.resize(frame2, (new_width2, output_height), interpolation=cv2.INTER_LINEAR)
        else:
            frame1 = cv2.resize(frame1, (new_width1, height1), interpolation=cv2.INTER_LINEAR)
            frame2 = cv2.resize(frame2, (new_width2, height2), interpolation=cv2.INTER_LINEAR)

        # Stack the frames horizontally
        combined_frame = cv2.hconcat([frame1, frame2])

        # Write the combined frame to the output video
        out.write(combined_frame)

    # Release all resources
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
video1 = 'flow_data_video_new.mp4'  # Path to the first video file
video2 = 'flow_gt_video.mp4'  # Path to the second video file
output_video = 'merged_video_lbs.mp4'  # Path to the output video file

stack_videos_horizontally(video1, video2, output_video)
