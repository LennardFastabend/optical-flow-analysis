from data_reader.reader import reader
from output_generation.visualizer import visualizer
import data_analysis.optical_flow as opflow
import data_analysis.geometric_quantification as geoquant
from data_segmentation.segmentation import Segmentation
from data_segmentation.segmentation import compute_intersection
from data_segmentation.segmentation import compute_left_border_intersection

import sys

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

root_dir = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis') #path to repository
input_dir = Path(r'data\PhaseContrastCleft\P01\input\Aligned\LinearStackAlignmentSift_Gauss5px.avi') #Read in Aligned Data! 
output_dir = Path(r'data\PhaseContrastCleft\P01\geoquant_filtered_images') 
input_reader = reader(root_dir, input_dir)
image_stack = input_reader.read_avi()

### Crop the image (resolves issues due to alignment of the images)
t, y, x = image_stack.shape
image_stack = image_stack[:, 100:y-100, 50:x-30] #crop the image

### calculate Example FlowFields for the defined time
dT=1
Tmax = 60
farneback_parameters = {"pyr_scale": 0.5,
                        "levels": 3,
                        "winsize": 5,#15,
                        "iterations": 3,
                        "poly_n": 5,
                        "poly_sigma": 1.2,
                        "flags": 0}
'''
print('Start Farneback Analysis')
flowfield_stack = opflow.FlowFieldStack(image_stack, farneback_parameters, t0=0, tfin=Tmax, dt=1)
print('Farneback Analysis Finished')
print()
'''

segmentation_generator = visualizer(root_dir, output_dir/Path('segmentation'))
#flowfield_generator = visualizer(root_dir, output_dir/Path('flowfields'))
#defmap_generator = visualizer(root_dir, output_dir/Path('defmap'))
#geoquant_generator = visualizer(root_dir, output_dir/Path('ComponentGeoQuant'))

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

T0=50
step = 10
temp_scale = 10

for T in np.arange(T0,T0+temp_scale,step):
    print(T)

    image = image_stack[T,...]
    #meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
    #defmap = opflow.calculateMagnitude(meanflowfield)

    # perform the segmentation
    cleft_mask, cleft_contour, front_mask, front_contour, edge_lines = Segmentation(image, segmentation_parameters)

    line_intersection = compute_intersection(edge_lines)
    left_border_intersection1 = compute_left_border_intersection(edge_lines[0], image)
    left_border_intersection2 = compute_left_border_intersection(edge_lines[1], image)

    print(line_intersection)
    print(left_border_intersection1)
    print(left_border_intersection2)
