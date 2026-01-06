import os

from decord import VideoReader, cpu
from PIL import Image


def extract_frames(video_path: str, output_dir: str, frame_skip: int = 10):
    """Extract frames from video using decord (memory efficient)."""
    os.makedirs(output_dir, exist_ok=True)

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    print(f"Total frames: {total_frames}")

    saved_idx = 0
    for i in range(0, total_frames, frame_skip):
        frame = vr[i].asnumpy()  # Loads one frame at a time
        Image.fromarray(frame).save(f"{output_dir}/frame_{saved_idx:05d}.jpg")
        saved_idx += 1

    print(f"Saved {saved_idx} frames to {output_dir}")
    return output_dir


def main():
    video_path = "./DJI_20250918142046_0264_W.mp4"  # Or path to JPEG folder
    output_dir = "./output/video"
    extract_frames(video_path, output_dir)


if __name__ == "__main__":
    main()
