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
input_dir = Path(r'data\Testing\Input\BW_People.avi') #Read in Aligned Data!
output_dir = Path(r'data\Testing\BW_People')

input_reader = reader(root_dir, input_dir)
image_stack = input_reader.read_avi()

### Crop the image (resolves issues due to alignment of the images)
t, y, x = image_stack.shape
print(image_stack.shape)
image_stack = image_stack[0:110, ...]#100:y-100, 50:x-30] #crop the image
T_offset = 0


temporal_scale = [1,10,25,50,100]
spatial_scale = [5,15,25,50,100]
def_scale = [10,10,10,10,10]

for i in range(len(temporal_scale)):
    for j in range(len(spatial_scale)):
        dt_OptFlow = temporal_scale[i]
        dxy_OptFlow = spatial_scale[j]
        max_magnitude = def_scale[i]

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
        defmap = opflow.calculateMagnitude(flowfield)

        ###Generate Images
        image_generator = visualizer(root_dir, output_dir/Path('dt_'+str(dt_OptFlow)+'_dxy_'+str(dxy_OptFlow)))

        # Save initial and final Image
        image_generator.saveImage(image_stack[0,...], title='Cleft at Time: '+str(T_offset), filename='Image'+str(T_offset))
        image_generator.saveImage(image_stack[0+dt_OptFlow,...], title='Cleft at Time: '+str(T_offset+dt_OptFlow), filename='Image'+str(T_offset+dt_OptFlow))

        # Save a Warped image , where the calculated Flow Field is applied too the initial image
        warped_image = warp_image(image_stack[0,...], flowfield)
        image_generator.saveImage(warped_image, title='InitImg warped with FlowField: '+str(T_offset)+'-'+str(T_offset+dt_OptFlow), filename='WarpedImage'+str(T_offset))

        # Save Flow Field and DefMaps
        image_generator.saveDeformationMap(defmap, min=0, max=max_magnitude, title='DefMap at Time: '+str(T_offset)+'-'+str(T_offset+dt_OptFlow), filename='DefMap'+str(T_offset))
        image_generator.saveDeformationMapRGB(flowfield, max_magnitude, title='DefMap at Time: '+str(T_offset)+'-'+str(T_offset+dt_OptFlow), filename='DefMapRGB'+str(T_offset))
        image_generator.saveFlowField(image_stack[0,...], flowfield, title='FlowField at Time: '+str(T_offset)+'-'+str(T_offset+dt_OptFlow), filename='FlowField'+str(T_offset), step=15, epsilon=0)



















sys.exit
### Crop the image (resolves issues due to alignment of the images)
t, y, x = image_stack.shape
crop_size = 25
image_stack = image_stack[:, crop_size:y-crop_size, crop_size:x-crop_size]

### calculate Example FlowFields for the defined time
dT=1
Tmax = 35
max_magnitude = 10

farneback_parameters = {"pyr_scale": 0.5,
                        "levels": 3,
                        "winsize": 5,#15,
                        "iterations": 3,
                        "poly_n": 5,
                        "poly_sigma": 1.2,
                        "flags": 0}

print('Start Farneback Analysis')
flowfield_stack = opflow.FlowFieldStack(image_stack, farneback_parameters, t0=0, tfin=Tmax, dt=1)
print('Farneback Analysis Finished')
print()

'''
image_generator = visualizer(root_dir, output_dir/Path('images'))
flowfield_generator = visualizer(root_dir, output_dir/Path('flowfields'))
defmap_generator = visualizer(root_dir, output_dir/Path('defmap'))
defmap_rgb_generator = visualizer(root_dir, output_dir/Path('defmap_rgb'))
#segmentation_generator = visualizer(root_dir, output_dir/Path('segmentation'))
#geoquant_generator = visualizer(root_dir, output_dir/Path('geometric_quantification'))


#save the color map for the rgb_defmap seperately
defmap_rgb_generator.saveHSVcolormap(max_magnitude, filename='colormap')
### T-Scan
for T in np.arange(0,Tmax):
    print('Output Time:', T)
    image = image_stack[T,...]
    #front_mask,filtered_image_front,front_contour_line = SegementGrowthFront(image)
    #cleft_mask,filtered_image_cleft,cleft_contour_line = SegementCleft(image)
    #segmentation_generator.saveSegmentationMasks(image, front_contour_line, cleft_contour_line, title='Segmentation at Time:'+str(T), filename='segmentation'+str(T))

    meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
    defmap = opflow.calculateMagnitude(meanflowfield)

    image_generator.saveImage(image_stack[T,...], title='Cleft at Time: '+str(T), filename='Image'+str(T))
    defmap_generator.saveDeformationMap(defmap, min=0, max=max_magnitude, title='DefMap at Time: '+str(T)+'-'+str(T+dT), filename='DefMap'+str(T))
    defmap_rgb_generator.saveDeformationMapRGB(meanflowfield, max_magnitude, title='DefMap at Time: '+str(T)+'-'+str(T+dT), filename='DefMap'+str(T))
    flowfield_generator.saveFlowField(image_stack[T,...], meanflowfield, title='FlowField at Time: '+str(T)+'-'+str(T+dT), filename='FlowField'+str(T), step=15, epsilon=0)
'''
########################################
### Example Spatio-Temporal Blurring ###
########################################
'''
defmap_generator = visualizer(root_dir, output_dir/Path('spatiotemporal_scales'))

T = 25
dT_array = [1,3,5,10,15]
kernel_sizes = [0,5,15,25,45]

for dT in dT_array:
    for kernel_size in kernel_sizes:
        meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
        smoothedmeanflowfield = opflow.BlurFlowField(meanflowfield,kernel_size)

        defmap = opflow.calculateMagnitude(smoothedmeanflowfield)

        defmap_generator.saveDeformationMap(defmap, min=0, max=max_magnitude, title='DefMap: dT='+str(dT)+'; Kernelsize='+ str(kernel_size), filename='DefMap_dT'+str(dT)+'_ksize'+str(kernel_size))
'''


'''
div_generator = visualizer(root_dir, output_dir/Path('divergence'))

T = 25
dT = 1

meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
div = opflow.Divergence(meanflowfield)

div_generator.saveDivergence(div, title='Divergence at Time: '+str(T)+'-'+str(T+dT), filename='Divergence'+str(T))
'''
