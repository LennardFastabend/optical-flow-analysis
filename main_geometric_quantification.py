from data_reader.reader import reader
from output_generation.visualizer import visualizer
import data_analysis.optical_flow as opflow
import data_analysis.geometric_quantification as geoquant
from data_segmentation.segmentation import Segmentation

import sys

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

root_dir = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis') #path to repository
input_dir = Path(r'data\PhaseContrastCleft\P01\input\Aligned\LinearStackAlignmentSift_Gauss5px.avi') #Read in Aligned Data! 
output_dir = Path(r'data\PhaseContrastCleft\P01\new_geometric_quantification')
input_reader = reader(root_dir, input_dir)
image_stack = input_reader.read_avi()

### Crop the image (resolves issues due to alignment of the images)
t, y, x = image_stack.shape
image_stack = image_stack[:, 100:y-100, 50:x-30] #crop the image

### calculate Example FlowFields for the defined time
dT=3
Tmax = 300

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

segmentation_generator = visualizer(root_dir, output_dir/Path('segmentation'))
#flowfield_generator = visualizer(root_dir, output_dir/Path('flowfields'))
defmap_generator = visualizer(root_dir, output_dir/Path('defmap'))
geoquant_generator = visualizer(root_dir, output_dir/Path('geometric_quantification'))

# define segmentation parameters
segmentation_parameters = { "cleft_gauss_ksize": 45,
                            "cleft_gauss_sigma": 5,
                            "cleft_canny_th1": 0,
                            "cleft_canny_th2": 45,
                            "cleft_hough_th": 125,
                            "front_sobel_ksize": 3,
                            "front_gauss_ksize": 3,
                            "front_gauss_sigma": 3,
                            "front_segmentation_th": 10,
                            "front_masksmoothing_ksize": 25,
                            "front_erosion_ksize": 3,
                            "front_erosion_iters": 3}
# paramters for geoquant visualisation
bin_size = 20
max_shown_distance = 1000
max_shown_displacement = 15
for T in np.arange(0,Tmax-dT,10):

    image = image_stack[T,...]

    # Calculate a Deformation Map
    meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
    defmap = opflow.calculateMagnitude(meanflowfield)

    # perform the segmentation
    cleft_mask, cleft_contour, front_mask, front_contour = Segmentation(image, segmentation_parameters)

    # find max x coordinate of growth front
    xmax_front = np.argmax(front_contour[:, 0])
    #print('xmax of growth front:', xmax_front)

    # define tissue region based on masks
    tissue_mask = cv2.subtract(cleft_mask, front_mask)

    # quantify deformation as a function of distance ti the growth front
    df = geoquant.GeometricQuantification(defmap, tissue_mask, front_contour, xmax_front, dx=100)

    title = 'Displacement vs. Distance to Growth Front at Time: '+ str(T) + '-' + str(T+dT)
    filename = 'geoquant' + str(T)
    geoquant_generator.saveGeometricQuantificationBinnedStatistics(df, bin_size, max_shown_distance, max_shown_displacement, title, filename+'binned')
    #geoquant_generator.saveGeometricQuantificationScatterPlot(df, max_shown_distance, max_shown_displacement, title, filename)
    defmap_generator.saveDeformationMap(defmap, min=0, max=10, title='DefMap at Time: '+str(T)+'-'+str(T+dT), filename='DefMap'+str(T))
    segmentation_generator.saveSegmentationMasks(image, front_contour, cleft_contour, title='Segmentation at Time:'+str(T), filename='segmentation'+str(T))
    #flowfield_generator.saveFlowField(image_stack[T,...], meanflowfield, title='FlowField at Time: '+str(T)+'-'+str(T+dT), filename='FlowField'+str(T), step=20, epsilon=0)



























###########################################################################
'''
print(df)
print('Max Distance:', df['distance'].max())
print('Min Distance:', df['distance'].min())
print('Max Displacement:', df['displacement'].max())
print('Min Displacement:', df['displacement'].min())
'''

'''
# Only consider deformations in he area of the tissue-mask
masked_defmap = defmap * tissue_mask/255

plt.imshow(masked_defmap, cmap='plasma', vmin=0, vmax=10)

plt.plot(front_contour_line[:, 0], front_contour_line[:, 1], marker='.', markersize=1, color='red', linestyle='-', linewidth=1)
plt.plot(cleft_contour_line[:, 0], cleft_contour_line[:, 1], marker='.', markersize=1, color='orange', linestyle='-', linewidth=1)
plt.plot(front_center[0], front_center[1], "og", markersize=5)
filename = 'MaskedDefMap'+str(T)
plt.savefig(root_dir / output_dir / filename, dpi=600)   # save the figure to file
plt.close()    # close the figure window

'''