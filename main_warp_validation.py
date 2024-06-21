import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_augmentation.warper import create_smooth_radial_deformation_field
from data_augmentation.warper import create_anisotropic_deformation_field
from data_augmentation.warper import create_random_deformation_field
from data_augmentation.warper import warp_image

from output_generation.visualizer import visualizer
import data_analysis.optical_flow as opflow


# Load the image
image = cv2.imread('data\PhaseContrastCleft\P01\input\P08#39_live_W03-P01_example_image.tif', cv2.IMREAD_GRAYSCALE)

# Define the center, radius, and strength of the deformation
center = (image.shape[1] // 2, image.shape[0] // 2)  # Center of the image
radius = 100#min(image.shape[0], image.shape[1]) // 4    # Radius of the deformation area
strength = 10                                  # Strength of the deformation

# Create the deformation field
#deformation_field = create_smooth_radial_deformation_field(image.shape, center, radius, strength)

deformation_field = create_random_deformation_field(image.shape, strength, num_blurs=25, blur_scale_range=(0.001, 0.01))

#direction = 'horizontal'  # Choose 'horizontal' or 'vertical'
#deformation_field = create_anisotropic_deformation_field(image.shape, center, radius, strength, direction)



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
calculated_deformation_field = flowfield_stack[0,...]

#'''
# Display the x components of the computed and GT Deformation field
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.title('Ground Truth X-Component')
max_abs_value = np.max(np.abs(-deformation_field[..., 0]))
plt.imshow(-deformation_field[...,0], cmap='gray', vmin=-max_abs_value, vmax=max_abs_value)
plt.colorbar()

plt.subplot(2, 2, 2)
plt.title('Calculated X-Component')
#max_abs_value = np.max(np.abs(calculated_deformation_field[..., 0]))
plt.imshow(calculated_deformation_field[...,0], cmap='gray', vmin=-max_abs_value, vmax=max_abs_value)
plt.colorbar()

plt.subplot(2, 2, 3)
plt.title('Ground Truth Y-Component')
max_abs_value = np.max(np.abs(calculated_deformation_field[..., 1]))
plt.imshow(-deformation_field[...,1], cmap='gray', vmin=-max_abs_value, vmax=max_abs_value)
plt.colorbar()

plt.subplot(2, 2, 4)
plt.title('Calculated Y-Component')
#max_abs_value = np.max(np.abs(calculated_deformation_field[..., 1]))
plt.imshow(calculated_deformation_field[...,1], cmap='gray', vmin=-max_abs_value, vmax=max_abs_value)
plt.colorbar()

plt.savefig(output_dir / 'DeformationComponents', dpi=600)   # save the figure to file
plt.close()
#'''
####################################################
field_differenz = calculated_deformation_field + deformation_field

plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.title('Field Differenz X-Component')
max_abs_value = np.max(np.abs(field_differenz[..., 0]))
plt.imshow(field_differenz[...,0], cmap='gray', vmin=-max_abs_value, vmax=max_abs_value)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title('Field Differenz Y-Component')
max_abs_value = np.max(np.abs(field_differenz[..., 1]))
plt.imshow(field_differenz[...,1], cmap='gray', vmin=-max_abs_value, vmax=max_abs_value)
plt.colorbar()
plt.savefig(output_dir / 'DeformationDifference', dpi=600)   # save the figure to file
plt.close()

#'''
output_generator = visualizer(root_dir, output_dir)
output_generator.saveFlowField(warped_image, calculated_deformation_field, title='Calculated FlowField', filename='CalculatedFlowField', step=10, epsilon=0)
output_generator.saveFlowField(warped_image, -deformation_field, title='Ground Truth FlowField', filename='GroundTruthFlowField', step=10, epsilon=0)
output_generator.saveImage(image, title='Image', filename='Image')
output_generator.saveImage(warped_image, title='Warped Image', filename='WarpedImage')
#'''
