import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_augmentation.warper import create_smooth_radial_deformation_field
from data_augmentation.warper import warp_image

from output_generation.visualizer import visualizer
import data_analysis.optical_flow as opflow


# Load the image
image = cv2.imread('data\PhaseContrastCleft\P01\input\P08#39_live_W03-P01_example_image.tif', cv2.IMREAD_GRAYSCALE)

# Define the center, radius, and strength of the deformation
center = (400, 400)#(image.shape[1] // 2, image.shape[0] // 2)  # Center of the image
radius = 100#min(image.shape[0], image.shape[1]) // 4    # Radius of the deformation area
strength = 150                                       # Strength of the deformation

# Create the radial deformation field
deformation_field = create_smooth_radial_deformation_field(image.shape, center, radius, strength)

# Warp the image using the deformation field
warped_image = warp_image(image, deformation_field)

'''
# Display the original and warped images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Warped Image')
plt.imshow(warped_image, cmap='gray')
plt.show()
'''

root_dir = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis') #path to repository
output_dir = Path(r'data\PhaseContrastCleft\P01\warp_validation')

stacked_images = np.stack([image, warped_image], axis=0)
farneback_parameters = {"pyr_scale": 0.5,
                        "levels": 3,
                        "winsize": 5,#15,
                        "iterations": 3,
                        "poly_n": 5,
                        "poly_sigma": 1.2,
                        "flags": 0}

print('Start Farneback Analysis')
flowfield_stack = opflow.FlowFieldStack(stacked_images, farneback_parameters, t0=0, tfin=1, dt=1)
print('Farneback Analysis Finished')

flowfield_generator = visualizer(root_dir, output_dir/Path('flowfields'))
flowfield_generator.saveFlowField(warped_image, flowfield_stack[0,...], title='Test', filename='Warptest', step=5, epsilon=0)