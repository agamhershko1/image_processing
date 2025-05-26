import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift

##############################################################################################################
# Question 1
##############################################################################################################
DEFAULT_LAYERS_NUM = 6
MAX_BRIGHTNESS = 255


def generate_gaussian_pyramid(image: np.ndarray, layers: int = DEFAULT_LAYERS_NUM) -> list:
    """
    Generate a Gaussian pyramid for a given image.
    :param image: The input image (numpy.ndarray)
    :param layers: The number of pyramid layers to generate.
    :return: A list containing the Gaussian pyramid levels.
    """
    pyramid = [image]
    for _ in range(layers):
        image = cv2.pyrDown(image)  # Downsample the image
        pyramid.append(image)
    return pyramid


def generate_laplacian_pyramid(gaussian_pyramid: list) -> list:
    """
    Generate a Laplacian pyramid from a given Gaussian pyramid.
    :param gaussian_pyramid: The Gaussian pyramid
    :return: A list containing the Laplacian pyramid levels.
    """
    laplacian_pyramid = [gaussian_pyramid[-1]]  # Start with the smallest level
    for i in range(len(gaussian_pyramid) - 1, 0, -1):
        gaussian_expended = cv2.resize(gaussian_pyramid[i],
                                       (gaussian_pyramid[i - 1].shape[1],
                                        gaussian_pyramid[i - 1].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expended)  # Subtract to get the details
        laplacian_pyramid.append(laplacian)
    return laplacian_pyramid


def combine_pyramids(lp_first: list, lp_second: list, gp_mask: list) -> list:
    """
    Combine two Laplacian pyramids using a Gaussian mask pyramid.
    :param lp_first: Laplacian pyramid of the first image
    :param lp_second: Laplacian pyramid of the second image
    :param gp_mask: Gaussian pyramid of the mask
    :return: A combined Laplacian pyramid
    """
    combined_pyramid = []
    for first_lap, second_lap, mask in zip(lp_first, lp_second, gp_mask):
        # Resize the mask to match the current Laplacian level
        mask_resized = cv2.resize(mask, (first_lap.shape[1], first_lap.shape[0]))
        mask_resized = mask_resized / MAX_BRIGHTNESS  # Normalize mask to [0, 1] range

        # Blend using the mask
        blended = first_lap * mask_resized + second_lap * (1 - mask_resized)
        combined_pyramid.append(blended)
    return combined_pyramid


def reconstruct_from_pyramid(combined_pyramid: list) -> np.ndarray:
    """
    Reconstruct an image from a combined Laplacian pyramid
    :param combined_pyramid: The combined Laplacian pyramid
    :return: The reconstructed image
    """
    combined_image = combined_pyramid[0]
    for i in range(1, len(combined_pyramid)):
        combined_image = cv2.resize(combined_image,
                                    (combined_pyramid[i].shape[1], combined_pyramid[i].shape[0]))
        combined_image = cv2.add(combined_pyramid[i], combined_image)  # Add the details back
    return combined_image


def blend_images(first_path: str, second_path: str, mask_path: str, output_path: str,
                 layers: int = DEFAULT_LAYERS_NUM) -> None:
    """
    Blend two images using pyramidal blending
    :param first_path: Path to the first image
    :param second_path: Path to the second image
    :param mask_path: Path to the mask image
    :param output_path: Path to save the blended output image
    :param layers: Number of pyramid layers to use
    """
    # Read the images
    first = cv2.imread(first_path).astype(np.float32)  # Load the first image as float32
    second = cv2.imread(second_path).astype(np.float32)  # Load the second image as float32
    mask = cv2.imread(mask_path)  # Load the mask

    # Generate Gaussian pyramids
    gp_first = generate_gaussian_pyramid(first, layers)
    gp_second = generate_gaussian_pyramid(second, layers)
    gp_mask = generate_gaussian_pyramid(mask, layers)

    # Generate Laplacian pyramids
    lp_first = generate_laplacian_pyramid(gp_first)
    lp_second = generate_laplacian_pyramid(gp_second)

    # Combine pyramids and reconstruct the blended image
    combined_pyramid = combine_pyramids(lp_first, lp_second, gp_mask)
    combined_reconstruct = reconstruct_from_pyramid(combined_pyramid)

    # Save the blended image to the specified output path
    cv2.imwrite(output_path, combined_reconstruct)

    # Display the results
    cv2.imshow('first', first / MAX_BRIGHTNESS)
    cv2.imshow('second', second / MAX_BRIGHTNESS)
    cv2.imshow('mask', mask / MAX_BRIGHTNESS)
    cv2.imshow('blended image', combined_reconstruct / MAX_BRIGHTNESS)

    # Wait and clean up
    cv2.waitKey(0)
    cv2.destroyAllWindows()


##############################################################################################################
# Question 2
##############################################################################################################
def hybridize_images(first_path: str, second_path: str, gaussian_path: str, output_path: str) -> None:
    """
    Combines two images using hybrid image technique in color mode.

    This function reads two color images and a grayscale Gaussian filter, then creates a hybrid image
    by combining the high-frequency components of the first image with the low-frequency components
    of the second image in the frequency domain.

    :param first_path: Path to the first image
    :param second_path: Path to the second image
    :param gaussian_path: Path to the mask image
    :param output_path: Path to save the blended output image
    """
    # Read the images
    first = cv2.imread(first_path).astype(np.float32)
    second = cv2.imread(second_path).astype(np.float32)
    gaussian = cv2.imread(gaussian_path, cv2.IMREAD_GRAYSCALE)

    # Normalize the Gaussian filter mask to ensure its values are between 0 and 1
    gaussian = gaussian / MAX_BRIGHTNESS

    # Initialize an array for the hybrid image
    hybrid_image = np.zeros_like(first)

    # Process each color channel separately, in opencv color format is BGR
    for channel in range(len(first[0][0])):  # for channel in 3 color channels
        first_channel = first[:, :, channel]
        second_channel = second[:, :, channel]

        # Compute the Fourier Transforms of the images
        first_dft = fftshift(fft2(first_channel))
        second_dft = fftshift(fft2(second_channel))

        # Apply the Gaussian mask to the Fourier transforms and combine the two Fourier transforms
        first_dft_weighted = first_dft * gaussian
        second_dft_weighted = second_dft * (1 - gaussian)
        hybrid_dft = first_dft_weighted + second_dft_weighted

        # Inverse Fourier Transform to get the hybrid image in the spatial domain
        hybrid_channel = np.abs(ifft2(hybrid_dft))

        # Store the result in the hybrid image array
        hybrid_image[:, :, channel] = hybrid_channel

    # Save the hybrid image to the specified output path
    cv2.imwrite(output_path, hybrid_image)

    # Display the results
    cv2.imshow('First Image', first / MAX_BRIGHTNESS)
    cv2.imshow('Second Image', second / MAX_BRIGHTNESS)
    cv2.imshow('Gaussian Filter', gaussian)
    cv2.imshow('Hybrid Image', hybrid_image / MAX_BRIGHTNESS)

    # Wait and clean up
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Question 1
    blend_images('part1/zion.png',
                 'part1/moon.png',
                 'part1/moon_mask.png',
                 'part1/blended_image.png')
    # Question 2
    hybridize_images('part2/Bennett.png',
                     'part2/Lapid.png',
                     'part2/gaussian.png',
                     'part2/hybrid_image.png')
