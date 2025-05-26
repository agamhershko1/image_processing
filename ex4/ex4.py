import cv2 as cv
import numpy as np
import imageio
import sys

INITIAL_INLIERS_GUESS = 0.5
RANSAC_PROBABILITY = 0.99
MAX_RANSAC_ITERATIONS = 100
RANSAC_DISTANCE_THRESHOLD = 1
RESPONSE_MAX_RATIO = 0.45
MIN_STRIP_WIDTH = 1
VIEWPOINT_SIZE = 11
FRAMES_PER_SECOND = 8
SHIFTS_NUMBER = 30
SHIFTING_CONSTANT_RATIO = 10


def get_frames_from_video(video_path, rotate=False):
    """
        Extract frames from a video file.

        Args:
        - video_path (str): Path to the video file.
        - rotate (bool): Flag to rotate frames, defaults to False.

        Returns:
        - frames (list of numpy arrays): List of frames from the video.
    """
    cap = cv.VideoCapture(video_path)  # Open the video file

    # Check if the video file was successfully opened
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Reads frames from the video capture and appends them to list until no more frames are available
    frames = []
    ret, frame = cap.read()
    while ret:
        frames.append(frame)
        ret, frame = cap.read()

    cap.release()  # Release the VideoCapture object to free system resources

    if rotate:
        frames = rotate_video(frames, True)  # Rotate frames if needed

    return frames


def rotate_video(frames, is_clockwise):
    """
    Rotates all frames by 90 degrees clockwise (up-down motion scenario).
    """
    if is_clockwise:
        direction = cv.ROTATE_90_CLOCKWISE
    else:
        direction = cv.ROTATE_90_COUNTERCLOCKWISE

    return [cv.rotate(frame, direction) for frame in frames]


def calculate_rigid_transformation(src_pts, dst_pts):
    """
    Calculate a rigid transformation (rotation + translation) between two sets of points.

    Args:
    - src_pts (numpy array): Source points, shape (N, 2).
    - dst_pts (numpy array): Destination points, shape (N, 2).

    Returns:
    - rigid_matrix (numpy array): A 3x3 rigid transformation matrix.
    """
    # Calculate the difference in x and y coordinates for the source points
    src_x_diff = src_pts[1][0] - src_pts[0][0]
    src_y_diff = src_pts[1][1] - src_pts[0][1]

    # Calculate the difference in x and y coordinates for the destination points
    dst_x_diff = dst_pts[1][0] - dst_pts[0][0]
    dst_y_diff = dst_pts[1][1] - dst_pts[0][1]

    # Compute the angle of the source and destination points and calculate the difference in rotation angles
    src_angle = np.arctan2(src_y_diff, src_x_diff)
    dst_angle = np.arctan2(dst_y_diff, dst_x_diff)
    rotation_angle = dst_angle - src_angle

    # Compute the rotation matrix, apply it to the first source point, and find the translation vector.
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])
    src_rotated = src_pts[0] @ rotation_matrix
    translation_vector = dst_pts[0] - src_rotated

    # Create a 3x3 matrix, insert the rotation and translation, and return the rigid matrix.
    rigid_matrix = np.eye(3)
    rigid_matrix[:2, :2] = rotation_matrix
    rigid_matrix[:2, 2] = translation_vector
    return rigid_matrix


def ransac_rigid(src_pts, dst_pts):
    """
       Perform RANSAC to estimate a rigid transformation between two sets of points.

       Args:
       - src_pts (numpy array): Source points, shape (N, 2).
       - dst_pts (numpy array): Destination points, shape (N, 2).
       - INITIAL_INLIERS_GUESS (float): Initial guess for the inlier ratio.
       - RANSAC_PROBABILITY (float): Probability that the algorithm will find the best transformation.
       - RANSAC_DISTANCE_THRESHOLD (float): Maximum allowed distance for a point to be considered an inlier.
       - MAX_RANSAC_ITERATIONS (int): Maximum number of iterations for RANSAC.

       Returns:
       - best_transformation_matrix (numpy array): The best rigid transformation matrix found.
    """
    # Initialize variables for points, inliers, and transformation.
    num_points = len(src_pts)
    min_samples = 2
    best_inliers_count = 0
    best_transformation_matrix = None

    # Set initial inliers guess, calculate number of iterations for RANSAC, and initialize iteration count.
    omega = INITIAL_INLIERS_GUESS
    iterations_num = int(np.log(1 - RANSAC_PROBABILITY) / np.log(1 - omega ** min_samples))
    iterations_count = 0

    # Run RANSAC iterations until the number of iterations exceeds the maximum or the calculated number
    while iterations_count < min(MAX_RANSAC_ITERATIONS, iterations_num):
        # Randomly select a subset of points (min_samples) from the source points
        indices = np.random.choice(num_points, min_samples, replace=False)
        src_subset = src_pts[indices]
        dst_subset = dst_pts[indices]

        rigid_matrix = calculate_rigid_transformation(src_subset, dst_subset)

        # Convert source points to homogeneous coordinates and apply the rigid transformation
        src_pts_homogeneous = np.hstack((src_pts, np.ones((num_points, 1))))
        transformed_src_pts = (rigid_matrix @ src_pts_homogeneous.T).T[:, :2]

        # Calculate distances between transformed source points and destination points
        distances = np.linalg.norm(transformed_src_pts - dst_pts, axis=1)

        # Identify inliers based on whether their distance is below the threshold
        inliers = distances < RANSAC_DISTANCE_THRESHOLD
        inliers_count = np.sum(inliers)

        # Update the best transformation if the current subset has more inliers
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_transformation_matrix = rigid_matrix

            # Adjust the inlier ratio and re-calculate the required iterations for RANSAC
            omega = inliers_count / num_points
            if 0 < omega < 1:
                iterations_num = int(np.log(1 - RANSAC_PROBABILITY) / np.log(1 - omega ** min_samples))
            else:
                iterations_num = MAX_RANSAC_ITERATIONS

        iterations_count += 1

    return best_transformation_matrix


def filter_keypoints_and_descriptors(keypoints, descriptors, response_threshold):
    """
    Filter keypoints and their corresponding descriptors based on response values.

    Args:
    - keypoints (list of cv2.KeyPoint): List of keypoints detected in an image.
    - descriptors (numpy array): Descriptors corresponding to the keypoints.
    - response_threshold (float): Threshold to filter keypoints based on their response value.

    Returns:
    - filtered_kps (list of cv2.KeyPoint): List of keypoints with response greater than the threshold.
    - filtered_descriptors (numpy array): Descriptors corresponding to the filtered keypoints.
    """

    # Filter keypoints based on their response values (keypoints with response greater than the threshold)
    filtered_kps = [kp for kp in keypoints if kp.response > response_threshold]

    # Filter the corresponding descriptors based on the selected keypoints
    filtered_descriptors = descriptors[
        [i for i, kp in enumerate(keypoints) if kp.response > response_threshold]]

    return filtered_kps, filtered_descriptors


def calculate_transformations(frames):
    """
    Calculate transformations between consecutive frames using SIFT and RANSAC.

    Args:
    - frames (list of numpy arrays): List of frames (images).

    Returns:
    - transformations (list of numpy arrays): List of rigid transformation matrices between consecutive frames
    """
    sift = cv.SIFT_create()
    transformations = []

    # Process each consecutive frame pair
    for i in range(1, len(frames)):
        previous_frame = frames[i - 1]
        current_frame = frames[i]

        # Detect keypoints and descriptors for both frames
        src_keypoints, src_descriptors = sift.detectAndCompute(previous_frame, None)
        dst_keypoints, dst_descriptors = sift.detectAndCompute(current_frame, None)

        # Calculate response thresholds for filtering keypoints
        src_response_threshold = RESPONSE_MAX_RATIO * np.max([kp.response for kp in src_keypoints])
        dst_response_threshold = RESPONSE_MAX_RATIO * np.max([kp.response for kp in dst_keypoints])

        # Filter keypoints based on response threshold
        src_keypoints, src_descriptors = filter_keypoints_and_descriptors(src_keypoints, src_descriptors,
                                                                          src_response_threshold)
        dst_keypoints, dst_descriptors = filter_keypoints_and_descriptors(dst_keypoints, dst_descriptors,
                                                                          dst_response_threshold)

        # Match descriptors using brute-force matcher (The matcher enforces a mutual consistency check)
        bf_matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = bf_matcher.match(src_descriptors, dst_descriptors)

        # Extract matched points
        src_pts = np.float32([src_keypoints[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([dst_keypoints[m.trainIdx].pt for m in matches])

        # Compute rigid transformation matrix with RANSAC
        rigid_matrix = ransac_rigid(src_pts, dst_pts)
        while rigid_matrix is None:
            rigid_matrix = ransac_rigid(src_pts, dst_pts)
        transformations.append(rigid_matrix)

    return transformations


def calculate_relative_transformations(transformations):
    """
        Calculate the relative transformations between frames based on a list of absolute transformations.

        Args:
        - transformations (list of numpy arrays): List of 3x3 transformation matrices (absolute
        transformations).

        Returns:
        - relative_transformations (list of numpy arrays): List of relative transformation matrices between
        consecutive frames.
    """
    relative_transformation = np.eye(3)  # Define Identity matrix for the first frame
    relative_transformations = [relative_transformation]

    for transformation in transformations:
        # Update the relative transformation by combining with the inverse
        inverse_transformation = np.linalg.inv(transformation)
        relative_transformation = relative_transformation @ inverse_transformation
        relative_transformations.append(relative_transformation)

    return relative_transformations


def calculate_frames_centers(frames, relative_transformations):
    """
        Calculate the transformed centers for frames based on relative transformations.

        Args:
        - frames (list of numpy arrays): List of frames (images).
        - relative_transformations (list of numpy arrays): List of 3x3 perspective transformation matrices.

        Returns:
        - frame_centers (list of tuples): List of transformed center coordinates (x, y) for each frame.
    """
    frame_height, frame_width = frames[0].shape[:2]
    center = np.float32([[[frame_width / 2, frame_height / 2]]])

    # For every frame, apply transformation on positions
    frame_centers = []

    for transform in relative_transformations:
        transformed_position = cv.transform(center, transform)
        frame_centers.append(tuple(map(int, transformed_position[0, 0, :])))

    return frame_centers


def calculate_viewpoints_positions(frames, relative_transformations):
    """
       Calculate the transformed positions for viewpoints across frames.

       Args:
       - frames (list of numpy arrays): List of frames (images).
       - relative_transformations (list of numpy arrays): List of 3x3 perspective transformations.
       - VIEWPOINT_SIZE (int): The number of viewpoints to calculate.

       Returns:
       - viewpoints_positions (list of lists): List of transformed positions for each viewpoint across frames.
    """
    # Find positions of frame
    frame_height = frames[0].shape[0]
    frame_width = frames[0].shape[1]

    viewpoints_positions = []
    for i in range(2, VIEWPOINT_SIZE - 1):
        # Vertical Center with changed horizontal positions
        positions = np.float32([[[frame_width * i / VIEWPOINT_SIZE, frame_height / 2]]])

        # For every frame, apply transformation on positions
        frame_positions = []
        for transform in relative_transformations:
            transformed_position = cv.transform(positions, transform)
            frame_positions.append(tuple(map(int, transformed_position[0, 0, :])))

        viewpoints_positions.append(frame_positions)

    return viewpoints_positions


def calculate_strip_borders(frames, frames_positions):
    """
       Calculate the borders for the strips between frames based on their center positions.

       Args:
       - frames (list of numpy arrays): List of frames.
       - frames_positions (list of tuples): List of the (x, y) positions of each frame.

       Returns:
       - strips_borders (list of lists): List of pairs representing the borders (min_x, max_x) of each strip.
    """
    strips_borders = []

    for i in range(len(frames) - 2):
        # Calculate the middle points between consecutive frames' centers
        min_x_strip = (frames_positions[i][0] + frames_positions[i + 1][0]) // 2
        max_x_strip = (frames_positions[i + 1][0] + frames_positions[i + 2][0]) // 2

        # If the direction is reversed (min_x > max_x), swap them
        if min_x_strip >= max_x_strip:
            min_x_strip, max_x_strip = max_x_strip, min_x_strip

        # Append the valid strip border between min_x_strip and max_x_strip
        strips_borders.append([min_x_strip, max_x_strip])

    return strips_borders


def calculate_canvas_sizes(frames, relative_transformations):
    """
       Calculate the canvas size and offsets to fit all transformed frames.

       Args:
       - frames (list of numpy arrays): List of frames (images).
       - relative_transformations (list of numpy arrays): List of 3x3 perspective transformations.

       Returns:
       - canvas_dimensions (tuple): Width and height of the canvas.
       - offsets (tuple): x and y offsets to align the frames.
   """
    frame_height = frames[0].shape[0]
    frame_width = frames[0].shape[1]

    # Vertical Center with changed horizontal position
    corners = np.float32([[[0, 0]], [[0, frame_height]], [[frame_width, frame_height]], [[frame_width, 0]]])

    # For every frame, apply transformation on corners
    transformed_corners = []
    for transform in relative_transformations:
        transformed_corners.append(cv.transform(corners, transform))

    # Compute min and max coordinates
    all_corners = np.vstack(transformed_corners).astype(int)
    x_min, y_min = np.min(all_corners[:, :, 0]), np.min(all_corners[:, :, 1])
    x_max, y_max = np.max(all_corners[:, :, 0]), np.max(all_corners[:, :, 1])

    # Define canvas size and offsets
    canvas_diemsions = (x_max - x_min, y_max - y_min)
    offsets = (-x_min, -y_min)

    return canvas_diemsions, offsets


def create_mosaic(frames, relative_transformations, canvas_sizes, strips_borders, shift, is_dynamic_mosaic):
    """
       Creates a mosaic from video frames by stitching strips based on transformations.

       Args:
           frames (list): List of video frames.
           relative_transformations (list): Relative transformations for stitching.
           canvas_sizes (tuple): Precomputed (canvas_dimensions, offsets).
           strips_borders (list): Borders defining the strip selection from each frame.
           shift (int): Amount of horizontal shift applied to strips.
           is_dynamic_mosaic (bool): Whether to crop dynamically based on strip positions.

       Returns:
           np.ndarray: The final mosaic image.
   """
    canvas_dimensions, offsets = canvas_sizes
    canvas_width, canvas_height = canvas_dimensions
    x_offset, y_offset = offsets
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    x_min, x_max = canvas_width, -canvas_width  # Initialize min/max x-coordinates for cropping

    for i, strip_borders in enumerate(strips_borders):
        # Ensure minimum width
        if strips_borders[i][0] == strips_borders[i][1]:
            strips_borders[i][1] += MIN_STRIP_WIDTH
        min_x_strip, max_x_strip = strip_borders

        # Apply shift
        min_x_strip += shift
        max_x_strip += shift

        # Update cropping boundaries
        x_min = np.min([x_min, min_x_strip])
        x_max = np.max([x_max, max_x_strip])

        # Warp and place the strip
        strip_warped = warp_strip(
            frames[i + 1], relative_transformations[i + 1], canvas_height, y_offset,
            (min_x_strip, max_x_strip))
        canvas[:, min_x_strip + x_offset:  max_x_strip + x_offset, :] = strip_warped

    # Crop dynamically if needed
    if is_dynamic_mosaic:
        canvas = canvas[:, x_min + x_offset:x_max + x_offset, :]

    return canvas


def create_dynamic_panoramas(frames, relative_transformations, canvas_sizes):
    """
       Creates a sequence of dynamic panoramas by shifting through the scene.

       Args:
           frames (list): List of input video frames.
           relative_transformations (list): Relative transformations between frames.
           canvas_sizes (list): Precomputed canvas sizes for panoramas.

       Returns:
           list: List of dynamically generated panorama frames.
   """
    # Compute center positions and strip borders
    frames_centers = calculate_frames_centers(frames, relative_transformations)
    strip_borders = calculate_strip_borders(frames, frames_centers)

    # Get frame width and compute shifting parameters
    frame_width = frames[0].shape[1]
    shift_size = frame_width // 2 - frame_width // SHIFTING_CONSTANT_RATIO

    # Generate dynamic panoramas
    dynamic_panoramas = []
    for shift in range(-shift_size, shift_size, shift_size // SHIFTS_NUMBER):
        dynamic_panorama = create_mosaic(frames, relative_transformations, canvas_sizes, strip_borders,
                                         shift, True)
        dynamic_panoramas.append(dynamic_panorama)

    return dynamic_panoramas


def create_viewpoints_panoramas(frames, relative_transformations, canvas_sizes):
    """
       Creates a sequence of panoramas from different viewpoints.

       Args:
           frames (list): List of frames from the video.
           relative_transformations (list): Relative transformations between frames.
           canvas_sizes (list): Precomputed canvas sizes for each panorama.

       Returns:
           list: List of viewpoint panorama frames.
   """
    viewpoints_positions = calculate_viewpoints_positions(frames, relative_transformations)

    # Create a list to store the mosaic frames
    viewpoint_frames = []
    for frames_positions in viewpoints_positions:
        strips_borders = calculate_strip_borders(frames, frames_positions)
        mosaic = create_mosaic(frames, relative_transformations, canvas_sizes, strips_borders,
                               0, False)
        viewpoint_frames.append(mosaic)

    # Add reversed frames to list
    return viewpoint_frames + viewpoint_frames[::-1]


def warp_strip(frame, transformation, canvas_height, y_offset, strip_borders):
    """
       Warps a vertical strip from the given frame using the provided transformation.

       Args:
           frame (numpy.ndarray): Input frame.
           transformation (numpy.ndarray): 3x3 homography matrix.
           canvas_height (int): Height of the output panorama canvas.
           y_offset (int): Vertical offset applied during warping.
           strip_borders (tuple): (x_min, x_max) defining the strip borders.

       Returns:
           numpy.ndarray: Warped strip of the frame.
   """
    strip_x_min, strip_x_max = strip_borders
    strip_width = strip_x_max - strip_x_min

    # Create a grid of coordinates
    x_coordinates, y_coordinates = np.meshgrid(np.arange(strip_x_min, strip_x_max),
                                               np.arange(-y_offset, canvas_height - y_offset))
    panorama_coordinates = np.stack(
        [x_coordinates.ravel(), y_coordinates.ravel(), np.ones_like(x_coordinates.ravel())], axis=-1).T

    # Backward warp panorama coordinates to the original frame
    inverse_transformation = np.linalg.inv(transformation)  # Inverse transformation
    frame_coordinates = inverse_transformation @ panorama_coordinates
    normalized_frame_coordinates = frame_coordinates / frame_coordinates[2]

    # Reshape the flattened x and y coordinates into 2D grid-like structures
    x_frame = normalized_frame_coordinates[0].reshape(canvas_height, strip_width).astype(np.float32)
    y_frame = normalized_frame_coordinates[1].reshape(canvas_height, strip_width).astype(np.float32)

    # Interpolate values of pixels using cv.remap
    strip_warped = cv.remap(frame, x_frame, y_frame,
                            interpolation=cv.INTER_LINEAR,
                            borderMode=cv.BORDER_CONSTANT,
                            borderValue=0)

    return strip_warped


def format_and_save_video(frames, is_dynamic, is_rotated, file_name):
    """
       Formats and saves a video from processed frames.

       Args:
           frames (list): List of frames (numpy arrays).
           is_dynamic (bool): Whether the video represents a dynamic panorama.
           is_rotated (bool): Whether the original video was rotated.
           file_name (str): Base name for the output file.

       Returns:
           str: Path to the saved video.
   """
    # Convert frames from BGR to RGB before saving and resizing to even sizes of frames
    frames_rgb = [cv.cvtColor(frame, cv.COLOR_BGR2RGB) for frame in frames]
    adjusted_frames = [adjust_frame_size(frame) for frame in frames_rgb]

    # Rotate frames back if necessary
    if is_rotated:
        adjusted_frames = rotate_video(adjusted_frames, False)

    # Set algorithm type
    algorithm = "dynamic" if is_dynamic else "viewpoint"

    # Save video
    output_video_path = "videos/" + algorithm + "_of_" + file_name + ".mp4"
    imageio.mimsave(output_video_path, adjusted_frames, fps=FRAMES_PER_SECOND, codec="libx264",
                    macro_block_size=1)
    print(f"Video saved as: {output_video_path}")


def adjust_frame_size(frame):
    """
        Ensures the frame dimensions are even by cropping if necessary.

        Args:
            frame (numpy.ndarray): Input image frame.

        Returns:
            numpy.ndarray: Cropped frame with even dimensions.
    """
    height, width = frame.shape[:2]

    # Ensure width and height are even by cropping 1 pixel if needed
    new_width = width if width % 2 == 0 else width - 1
    new_height = height if height % 2 == 0 else height - 1

    return cv.resize(frame, (new_width, new_height))  # Crop instead of resizing


def get_boolean_input(message):
    """
        Prompts the user for a boolean input and returns True or False.

        Args:
            message (str): The message displayed to the user.

        Returns:
            bool: True if the user inputs a positive response, False  if the user inputs a negative response.
    """
    while True:
        input_value = input(message)

        if input_value in ["True", "true", "TRUE", "1", "yes", "YES", "Yes"]:
            return True
        elif input_value in ["False", "false", "FALSE", "0", "NO", "no", "No"]:
            return False
        else:
            print("Invalid input. Please enter again")


if __name__ == "__main__":
    try:
        # Handle arguments
        video_path = file_path = sys.argv[1]
        file_name = file_path.split('/')[-1].split('.')[0]

        # Handle input
        is_video_rotated = get_boolean_input("Should video be rotated (from vertical to horizontal)? ")
        is_panorama_dynamic = get_boolean_input(
            "Is panorama dynamic? Else, a video of panoramas from different viewpoints will be created\n")

        # Extract frames from the video
        frames = get_frames_from_video(video_path, is_video_rotated)

        # Calculate transformations
        transformations = calculate_transformations(frames)
        relative_transformations = calculate_relative_transformations(transformations)
        canvas_sizes = calculate_canvas_sizes(frames, relative_transformations)

        # Generate panorama based on user choice
        if is_panorama_dynamic:
            mosaic_frames = create_dynamic_panoramas(frames, relative_transformations, canvas_sizes)
        else:
            mosaic_frames = create_viewpoints_panoramas(frames, relative_transformations, canvas_sizes)

        # Formats and saves the output video based on user specifications
        format_and_save_video(mosaic_frames, is_panorama_dynamic, is_video_rotated, file_name)
    except IOError as e:
        print(e)
    finally:
        cv.destroyAllWindows()
