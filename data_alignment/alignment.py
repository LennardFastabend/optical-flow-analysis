from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage import io
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

def align_image_stack(image_stack):
    t, y, x = image_stack.shape

    # Initialize an empty array to hold the registered images
    registered_stack = np.zeros_like(image_stack)

    # Copy the initial image of the image stack:
    registered_stack[0, :, :] = image_stack[0, :, :]

    # Iterate over each frame in the time series
    for i in range(1,t):
        print('Alignment at Time: ' + str(i))
        current_image = image_stack[i, :, :]
        reference_image = registered_stack[i-1, :, :]
        
        # Compute the translation needed to align current image with the reference
        shift_estimation, error, _ = phase_cross_correlation(reference_image, current_image)
        
        # Apply the estimated shift to the current image
        registered_image = shift(current_image, shift_estimation, mode='constant', cval=0)
        
        # Store the registered image in the stack
        registered_stack[i, :, :] = registered_image

    return registered_stack

def align_image_stack_to_reference(image_stack, reference_image):
    t, y, x = image_stack.shape

    # Initialize an empty array to hold the registered images
    registered_stack = np.zeros_like(image_stack)

    # Iterate over each frame in the time series
    for i in range(t):
        print('Alignment at Time: ' + str(i))
        current_image = image_stack[i, :, :]
        
        # Compute the translation needed to align current image with the reference
        shift_estimation, error, _ = phase_cross_correlation(reference_image, current_image)
        
        # Apply the estimated shift to the current image
        registered_image = shift(current_image, shift_estimation, mode='constant', cval=0)
        
        # Store the registered image in the stack
        registered_stack[i, :, :] = registered_image

    return registered_stack


def save_image_stack_video(image_stack, filename):
    #Save as a video using OpenCV
    out = cv2.VideoWriter('registered_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (x, y))

    for i in range(t):
        # Convert image to uint8 format
        frame = cv2.normalize(registered_stack[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    out.release()