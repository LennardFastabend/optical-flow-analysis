import cv2
import numpy as np

'''
Note: The deformation fields used to warp the image always show in the oposite direction of the apparent displacement
'''

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

def create_anisotropic_deformation_field(shape, center, radius, strength, direction='horizontal'):
    height, width = shape[:2]
    deformation_field = np.zeros((height, width, 2), dtype=np.float32)

    # Generate the grid of coordinates
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # Compute the distance from each pixel to the center
    distance_x = grid_x - center[0]
    distance_y = grid_y - center[1]
    distance = np.sqrt(distance_x**2 + distance_y**2)

    # Create the anisotropic deformation using a Gaussian-like function
    sigma = radius / 3  # Standard deviation for Gaussian function
    deformation_strength = np.exp(-(distance**2) / (2 * sigma**2)) * strength

    # Apply the anisotropic deformation based on the specified direction
    mask = distance < radius
    if direction == 'horizontal':
        deformation_field[..., 0] = mask * deformation_strength * distance_x / (radius + 1e-5)
        deformation_field[..., 1] = mask * deformation_strength * 0  # No vertical deformation
    elif direction == 'vertical':
        deformation_field[..., 0] = mask * deformation_strength * 0  # No horizontal deformation
        deformation_field[..., 1] = mask * deformation_strength * distance_y / (radius + 1e-5)
    else:
        raise ValueError("Invalid direction. Use 'horizontal' or 'vertical'.")

    return deformation_field

def create_random_deformation_field(shape, strength, num_blurs=3, blur_scale_range=(0.05, 0.2)):
    height, width = shape[:2]
    deformation_field = np.zeros((height, width, 2), dtype=np.float32)

    # Generate random displacement fields for X and Y
    random_x = np.random.uniform(-1, 1, (height, width)) * strength
    random_y = np.random.uniform(-1, 1, (height, width)) * strength

    # Randomly choose blur scales from the specified range
    blur_scales = np.random.uniform(blur_scale_range[0], blur_scale_range[1], num_blurs)

    # Apply multiple Gaussian blurs with randomly chosen scales
    for scale in blur_scales:
        sigma_x = width * scale
        sigma_y = height * scale
        random_x = cv2.GaussianBlur(random_x, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
        random_y = cv2.GaussianBlur(random_y, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)

    # Normalize the random fields
    random_x = (random_x - np.min(random_x)) / (np.max(random_x) - np.min(random_x)) * 2 - 1
    random_y = (random_y - np.min(random_y)) / (np.max(random_y) - np.min(random_y)) * 2 - 1

    # Apply the strength factor
    deformation_field[..., 0] = random_x * strength
    deformation_field[..., 1] = random_y * strength

    return deformation_field