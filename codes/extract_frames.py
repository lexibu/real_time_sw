import cv2
import os

# Path to the video file
video_path = "../data/phases_2024_fancy_2160p30.mp4"
video_name = os.path.basename(video_path)[: -len(".mp4")]
# Directory to save the frames
output_dir = "../data/frames"
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    # Stop the code execution
    # exit(1)

# Frame counter
frame_count = 0

# Read and save frames
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Construct the frame file name
    frame_filename = os.path.join(output_dir, f"{video_name}_{frame_count:05d}.png")

    # Save the frame as PNG
    cv2.imwrite(frame_filename, frame)

    print(f"Saved {frame_filename}")

    frame_count += 1

# Release the video capture object
cap.release()

print(f"Extracted {frame_count} frames from {video_path}")
