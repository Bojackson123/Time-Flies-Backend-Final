import time
from api.analyze_video import analyze_video
import cv2, os

def convert_video_to_frames(video_path, save_dir):
    # Make sure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Determine video FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        # Read a frame
        success, frame = cap.read()

        # If the frame was not successfully read, break the loop
        if not success:
            break

        # Save every frame without trying to simulate 30 FPS
        save_path = os.path.join(save_dir, f'frame_{frame_count:04d}.png')
        cv2.imwrite(save_path, frame)
        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f'All frames have been saved to {save_dir}. Total frames: {frame_count}.')
    return original_fps

def test_analyze_video(video_path):
    save_dir = 'test_frames'
    convert_video_to_frames(video_path, save_dir)
    
    start_time = time.time()  # Start timing
    figure, Json = analyze_video(save_dir)
    elapsed_time = time.time() - start_time  # End timing

    print(f"analyze_video function took {elapsed_time:.2f} seconds to execute.")
    return

test_analyze_video('test_video/test.mp4')