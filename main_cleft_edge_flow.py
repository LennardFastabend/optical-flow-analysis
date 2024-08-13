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
output_dir = Path(r'data\PhaseContrastCleft\P01\cleft_edge_flow') 
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
#'''
print('Start Farneback Analysis')
flowfield_stack = opflow.FlowFieldStack(image_stack, farneback_parameters, t0=0, tfin=Tmax, dt=1)
print('Farneback Analysis Finished')
print()
#'''

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
    meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
    defmap = opflow.calculateMagnitude(meanflowfield)

    # perform the segmentation
    cleft_mask, cleft_contour, front_mask, front_contour, edge_lines = Segmentation(image, segmentation_parameters)

    line_intersection = compute_intersection(edge_lines)
    left_border_intersection1 = compute_left_border_intersection(edge_lines[0], image)
    left_border_intersection2 = compute_left_border_intersection(edge_lines[1], image)

    #oder the border intersections to define a upper and lower cleft edge
    if left_border_intersection1[1] < left_border_intersection2[1]:
        upper_border_intersection = left_border_intersection1
        lower_border_intersection = left_border_intersection2
    else:
        upper_border_intersection = left_border_intersection2
        lower_border_intersection = left_border_intersection1

    #define vectors that point from the image boundaries to th eline intersection
    upper_line_vector = line_intersection - upper_border_intersection
    lower_line_vector = line_intersection - lower_border_intersection
    # Normalize the vectors to length 1
    upper_line_vector = upper_line_vector / np.linalg.norm(upper_line_vector)
    lower_line_vector = lower_line_vector / np.linalg.norm(lower_line_vector)

    #define orthogonal vectors to the lines, pointing inward to the cleft
    upper_normal_vector = np.array([-upper_line_vector[1], upper_line_vector[0]])/8 #rotate +90°
    lower_normal_vector = np.array([lower_line_vector[1], -lower_line_vector[0]])/8 #rotate -90°
    # Normalize the vectors to length 1
    upper_normal_vector = upper_normal_vector / np.linalg.norm(upper_normal_vector)
    lower_normal_vector = lower_normal_vector / np.linalg.norm(lower_normal_vector)

    print(lower_normal_vector)


    ### Example Analysis for lower cleft edge:
    #input:
    origin = lower_border_intersection
    #image

    ### Define a position vector p for each pixel, that points from the boundary intersection to the pixel (apply on the whole image)

    # Create a grid of pixel coordinates
    height, width = image.shape
    y, x = np.indices((height, width))  # y: row indices, x: column indices

    # Calculate vectors from the origin to each pixel
    vectors_x = x - origin[0]  # x-coordinates of vectors
    vectors_y = y - origin[1]  # y-coordinates of vectors

    # Combine the x and y components into a single array of shape (height, width, 2)
    positions = np.stack((vectors_x, vectors_y), axis=-1)

    pos_w = np.sum(positions * lower_normal_vector, axis=-1)
    pos_l = np.sum(positions * lower_line_vector, axis=-1)

    ### Store pos_l, pos_w and doformation values in a dataframe for visualisation
    #define max/min values for the l and w position (this defines the analysed ROI)
    # Define min and max values for l and w
    min_l, max_l = 50,1200
    min_w, max_w = -500,500

    # Create a mask for filtering based on the min and max values
    mask = (pos_l >= min_l) & (pos_l <= max_l) & (pos_w >= min_w) & (pos_w <= max_w)

    # Apply the mask to pos_l, pos_w, and deformation
    filtered_l = pos_l[mask]
    filtered_w = pos_w[mask]
    filtered_deformation = defmap[mask]

    # Plotting
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(filtered_l, filtered_w, c=filtered_deformation, cmap='viridis', s=10)
    plt.colorbar(sc, label='Deformation Value')
    plt.xlabel('l')
    plt.ylabel('w')
    plt.title('Deformation in l-w Coordinate System')
    plt.xlim(min_l, max_l)
    plt.ylim(min_w, max_w)
    plt.grid(True)
    plt.show()



    '''
    plt.imshow(image, cmap='gray')
    #plt.plot(line_intersection[0], line_intersection[1], marker='.', color='green')

    plt.plot(upper_border_intersection[0], upper_border_intersection[1], marker='.', color='orange')
    plt.arrow(upper_border_intersection[0], upper_border_intersection[1], upper_line_vector[0], upper_line_vector[1],head_width=2, head_length=3, fc='g', ec='g')
    plt.arrow(upper_border_intersection[0], upper_border_intersection[1], upper_normal_vector[0], upper_normal_vector[1],head_width=10, head_length=15, fc='r', ec='r')

    plt.plot(lower_border_intersection[0], lower_border_intersection[1], marker='.', color='red')
    plt.arrow(lower_border_intersection[0], lower_border_intersection[1], lower_line_vector[0], lower_line_vector[1],head_width=10, head_length=15, fc='g', ec='g')
    plt.arrow(lower_border_intersection[0], lower_border_intersection[1], lower_normal_vector[0], lower_normal_vector[1],head_width=10, head_length=15, fc='r', ec='r')
    plt.show()
    '''

    
