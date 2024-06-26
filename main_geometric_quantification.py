from data_reader.reader import reader
from output_generation.visualizer import visualizer
import data_analysis.optical_flow as opflow
import data_analysis.geometric_quantification as geoquant
from data_segmentation.segmentation import SegementCleft
from data_segmentation.segmentation import SegementGrowthFront

import sys

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

root_dir = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis') #path to repository
input_dir = Path(r'data\PhaseContrastCleft\P01\input\Aligned\LinearStackAlignmentSift_Gauss5px.avi') #Read in Aligned Data!
output_dir = Path(r'data\PhaseContrastCleft\P01\output_geometric_quantification')

input_reader = reader(root_dir, input_dir)
image_stack = input_reader.read_avi()

### Crop the image (resolves issues due to alignment of the images)
t, y, x = image_stack.shape
crop_size = 25
image_stack = image_stack[:, crop_size:y-crop_size, crop_size:x-crop_size]

### calculate Example FlowFields for the defined time
dT=3
Tmax = 100

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
flowfield_generator = visualizer(root_dir, output_dir/Path('flowfields'))
defmap_generator = visualizer(root_dir, output_dir/Path('defmap'))
geoquant_generator = visualizer(root_dir, output_dir/Path('geometric_quantification'))

### T-Scan
'''
for T in np.arange(0,Tmax):
    print('Output Time:', T)
    image = image_stack[T,...]
    front_mask,filtered_image_front,front_contour_line = SegementGrowthFront(image)
    cleft_mask,filtered_image_cleft,cleft_contour_line = SegementCleft(image)
    segmentation_generator.saveSegmentationMasks(image, front_contour_line, cleft_contour_line, title='Segmentation at Time:'+str(T), filename='segmentation'+str(T))

    meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
    defmap = opflow.calculateMagnitude(meanflowfield)
    flowfield_generator.saveFlowField(image_stack[T,...], meanflowfield, title='FlowField at Time: '+str(T)+'-'+str(T+dT), filename='FlowField'+str(T), step=20, epsilon=0)
    defmap_generator.saveDeformationMap(defmap, min=0, max=10, title='DefMap at Time: '+str(T)+'-'+str(T+dT), filename='DefMap'+str(T))

segmentation_generator.create_video(fps=10)
flowfield_generator.create_video(fps=10)
defmap_generator.create_video(fps=10)
'''
### Example Analysis for frame T
# get image
for T in np.arange(0,Tmax):
    image = image_stack[T,...]

    # Calculate a Deformation Map
    meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
    defmap = opflow.calculateMagnitude(meanflowfield)

    # define masks
    front_mask,filtered_image_front,front_contour_line = SegementGrowthFront(image)
    cleft_mask,filtered_image_cleft,cleft_contour_line = SegementCleft(image)

    # find center of growth front
    front_center = front_contour_line[np.argmax(front_contour_line[:, 0])]
    print('front center:', front_center)

    # define tissue region based on masks
    tissue_mask = cv2.subtract(cleft_mask, front_mask)

    # quantify deformation as a function of distance ti the growth front
    df = geoquant.GeometricQuantification(defmap, tissue_mask, front_contour_line, front_center, dx=100)

    bin_num=50
    title = 'Mean Displacement vs. Binned Distance at Time: '+str(T)
    filename = 'geoquant' + str(T)
    geoquant_generator.saveGeometricQuantification(df, bin_num, title, filename)



























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

plt.imshow(tissue_mask, cmap='gray')
plt.plot(front_center[0], front_center[1], "og", markersize=5)
plt.show()
'''