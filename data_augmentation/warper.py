import cv2
import numpy as np

def warp_image(image, deformation_field):
    # Ensure the deformation field has the same shape as the image
    if deformation_field.shape[:2] != image.shape[:2]:
        raise ValueError("Deformation field must have the same height and width as the image.")

    # Generate the grid of coordinates
    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # Apply the deformation field to the grid
    map_x = (grid_x + deformation_field[..., 0]).astype(np.float32)
    map_y = (grid_y + deformation_field[..., 1]).astype(np.float32)

    # Warp the image using the remap function
    warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return warped_image

def create_smooth_radial_deformation_field(shape, center, radius, strength):
    height, width = shape[:2]
    deformation_field = np.zeros((height, width, 2), dtype=np.float32)

    # Generate the grid of coordinates
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # Compute the distance from each pixel to the center
    distance_x = grid_x - center[0]
    distance_y = grid_y - center[1]
    distance = np.sqrt(distance_x**2 + distance_y**2)

    # Create the smooth radial deformation using a Gaussian-like function
    sigma = radius / 3  # Standard deviation for Gaussian function
    deformation_strength = np.exp(-(distance**2) / (2 * sigma**2)) * strength

    # Apply the deformation only within the specified radius
    mask = distance < radius
    deformation_field[..., 0] = mask * deformation_strength * distance_x / radius
    deformation_field[..., 1] = mask * deformation_strength * distance_y / radius

    return deformation_field