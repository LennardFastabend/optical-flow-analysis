from data_reader.reader import reader
from output_generation.visualizer import visualizer
import data_analysis.optical_flow as opflow
import data_analysis.geometric_quantification as geoquant
from data_segmentation.segmentation import Segmentation
from data_augmentation.warper import warp_image

import sys

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

root_dir = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis') #path to repository
input_dir = Path(r'data\PhaseContrastCleft\P01\input\Aligned\LinearStackAlignmentSift_Gauss5px.avi') #Read in Aligned Data!
output_dir = Path(r'data\PhaseContrastCleft\P01\Flow_Validation\T150')

input_reader = reader(root_dir, input_dir)
image_stack = input_reader.read_avi()

### Crop the image (resolves issues due to alignment of the images)
t, y, x = image_stack.shape
print(image_stack.shape)
image_stack = image_stack[150:260, 100:y-100, 50:x-30] #crop the image
T_offset = 150

image_generator = visualizer(root_dir, output_dir)

temporal_scale = np.arange(1,101,1)#[1,10,25,50,100] #
spatial_scale = np.arange(1,101,1)

#FB_error_list = []
#residual_error_list = []

# Initialize lists to store errors
FB_error_image = np.zeros((len(temporal_scale), len(spatial_scale)))  # 2D array for FB errors
residual_error_image = np.zeros((len(temporal_scale), len(spatial_scale)))  # 2D array for residual errors

for i in range(len(temporal_scale)):
        for j in range(len(temporal_scale)):
            dt_OptFlow = temporal_scale[i]
            dxy_OptFlow = spatial_scale[j]


            print('Perform Optical Flow Analysis with:')
            print('dt: ', dt_OptFlow)
            print('dxy: ', dxy_OptFlow)

            ### calculate Example FlowFields for the defined time
            Tmax = dt_OptFlow #only generate one Flow Field for each condition
            farneback_parameters = {"pyr_scale": 0.5,
                                    "levels": 3,
                                    "winsize": dxy_OptFlow,
                                    "iterations": 3,
                                    "poly_n": 5,
                                    "poly_sigma": 1.2,
                                    "flags": 0}
            #'''
            print('Start Farneback Analysis')
            flowfield_stack = opflow.FlowFieldStack(image_stack, farneback_parameters, t0=0, tfin=1, dt=dt_OptFlow)
            print(flowfield_stack.shape)
            print('Farneback Analysis Finished')
            print()
            #'''

            flowfield = flowfield_stack[0,...]
            backwards_flowfield = opflow.FlowField(image_stack[dt_OptFlow,...], image_stack[0,...], farneback_parameters)
            error_flowfield = flowfield-backwards_flowfield
            error_map = opflow.calculateMagnitude(error_flowfield)

            FB_error = np.sum(error_map)/(error_map.shape[0]*error_map.shape[1])
            #FB_error_list.append(FB_error)
            FB_error_image[i,j] = FB_error

            #image_generator.saveImage(error_map, title='Forward-Backward-Error with dt: '+str(dt_OptFlow), filename='FB_Error'+str(dt_OptFlow))
            '''
            plt.imshow(error_map, cmap='gray')#, vmin=0, vmax=1)
            plt.show()
            '''

            warped_image = warp_image(image_stack[0,...], flowfield)

            ### Residual-Image
            residual_image = np.sqrt((image_stack[dt_OptFlow,...]/255 - warped_image/255)**2)

            residual_error = np.sum(residual_image)/(residual_image.shape[0]*residual_image.shape[1])
            #residual_error_list.append(residual_error)
            residual_error_image[i, j] = residual_error

            # Visualise the normalized residual
            #image_generator.saveImage(residual_image, title='Residual-Image with dt: '+str(dt_OptFlow), filename='Residual'+str(dt_OptFlow))
            '''
            plt.imshow(residual_image, cmap='gray', vmin=0, vmax=1)
            plt.show()
            '''

# Plot FB_error_list
plt.figure(figsize=(10, 8))
plt.imshow(FB_error_image, extent=[spatial_scale.min(), spatial_scale.max(), temporal_scale.min(), temporal_scale.max()], origin='lower', aspect='auto', cmap='plasma')
plt.colorbar()
plt.title('Mean Forward-Backward-Error vs Temporal and Spatial Scale')
plt.xlabel('Spatial Scale')
plt.ylabel('Temporal Scale')
plt.xticks(np.linspace(spatial_scale.min(), spatial_scale.max(), num=10, dtype=int))
plt.yticks(np.linspace(temporal_scale.min(), temporal_scale.max(), num=10, dtype=int))
plt.show()

# Plot residual_error_list
plt.figure(figsize=(10, 8))
plt.imshow(residual_error_image, extent=[spatial_scale.min(), spatial_scale.max(), temporal_scale.min(), temporal_scale.max()], origin='lower', aspect='auto', cmap='plasma')
plt.colorbar()
plt.title('Mean Residual Error vs Temporal and Spatial Scale')
plt.xlabel('Spatial Scale')
plt.ylabel('Temporal Scale')
plt.xticks(np.linspace(spatial_scale.min(), spatial_scale.max(), num=10, dtype=int))
plt.yticks(np.linspace(temporal_scale.min(), temporal_scale.max(), num=10, dtype=int))
plt.show()



'''
# Create a scatter plot
plt.figure(figsize=(8, 5))
#plt.plot(temporal_scale, residual_error_list, marker='o', linestyle='', color='b', label='Error vs. dt_OptFlow')
plt.plot(temporal_scale, FB_error_list, marker='o', linestyle='', color='r', label='Error vs. dt_OptFlow')

# Add titles and labels
plt.title('Forward-Backward-Error per Pixel')
plt.xlabel('dt_OptFlow')
plt.ylabel('Forward-Backward-Error')
plt.grid()
#plt.legend()
#plt.xscale('log')  # Optional: use log scale if the range of dt is wide
#plt.yscale('log')  # Optional: use log scale if the range of errors is wide
plt.show()
'''

sys.exit()

###Generate Images
image_generator = visualizer(root_dir, output_dir/Path('dt_'+str(dt_OptFlow)+'_dxy_'+str(dxy_OptFlow)))

# Save initial and final Image
image_generator.saveImage(image_stack[0,...], title='Cleft at Time: '+str(T_offset), filename='Image'+str(T_offset))
image_generator.saveImage(image_stack[0+dt_OptFlow,...], title='Cleft at Time: '+str(T_offset+dt_OptFlow), filename='Image'+str(T_offset+dt_OptFlow))

# Save a Warped image , where the calculated Flow Field is applied too the initial image

image_generator.saveImage(warped_image, title='InitImg warped with FlowField: '+str(T_offset)+'-'+str(T_offset+dt_OptFlow), filename='WarpedImage'+str(T_offset))

# Save Flow Field and DefMaps
image_generator.saveDeformationMap(defmap, min=0, max=max_magnitude, title='DefMap at Time: '+str(T_offset)+'-'+str(T_offset+dt_OptFlow), filename='DefMap'+str(T_offset))
image_generator.saveDeformationMapRGB(flowfield, max_magnitude, title='DefMap at Time: '+str(T_offset)+'-'+str(T_offset+dt_OptFlow), filename='DefMapRGB'+str(T_offset))
image_generator.saveFlowField(image_stack[0,...], flowfield, title='FlowField at Time: '+str(T_offset)+'-'+str(T_offset+dt_OptFlow), filename='FlowField'+str(T_offset), step=15, epsilon=0)
