import mediapy as media
from PIL import Image as Image
import numpy as np
import sys

GRAYSCALE_CODE = "L"


def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected
     (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    video = np.array(media.read_video(video_path))
    hist_diff = []
    prev_sum = 0

    for i, frame in enumerate(video):
        gray_frame = np.array(Image.fromarray(frame).convert(GRAYSCALE_CODE))
        histogram = np.histogram(gray_frame.ravel(), bins=256, range=(0, 255))[0]
        cumulative_histogram = np.cumsum(histogram)
        current_sum = np.sum(cumulative_histogram)
        if i > 0:
            hist_diff.append(np.abs(current_sum - prev_sum))
        prev_sum = current_sum

    return int(np.argmax(hist_diff)), int(np.argmax(hist_diff) + 1)


if __name__ == '__main__':
    print(main(sys.argv[1], sys.argv[2]))
