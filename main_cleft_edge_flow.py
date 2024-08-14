from data_reader.reader import reader
from output_generation.visualizer import visualizer
import data_analysis.optical_flow as opflow
import data_analysis.cleft_edge_flow as cleft_edge_flow
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

    #order the border intersections to define a upper and lower cleft edge
    if left_border_intersection1[1] < left_border_intersection2[1]:
        upper_border_intersection = left_border_intersection1
        lower_border_intersection = left_border_intersection2
    else:
        upper_border_intersection = left_border_intersection2
        lower_border_intersection = left_border_intersection1

    ### define basis vectors for both cleft edges (line vector points in the direction of the line, normal vector points orthogonal into the cleft/tissue)
    upper_line_vector, upper_normal_vector, lower_line_vector, lower_normal_vector = cleft_edge_flow.normalized_line_and_normal_vectors(line_intersection, lower_border_intersection, upper_border_intersection)


    ##### Example Analysis for the lower cleft edge:
    ### The origin for each line is the respective border intersection of the line with the left image border
    lower_origin = lower_border_intersection
    ### Calculate Positions relative to new origin (return local positional component in line and width/normal direction)
    pos_l, pos_w = cleft_edge_flow.positional_vector_transformation(image, lower_origin, lower_line_vector, lower_normal_vector)


    ### calculate the parallel and normal deformation components relative to the cleft edge
    parallel_def, normal_def = cleft_edge_flow.deformation_projections(meanflowfield, lower_line_vector, lower_normal_vector)


    ### Visualize pos_l, pos_w and deformation values in a l-w-plot




    #define max/min values for the l and w position (this defines the visualized ROI)
    min_l, max_l = 100,1300 #length range in pixels from the origin
    min_w, max_w = -200,200 #width range in pixels from the origin (positive is inside the cleft, negative outside)

    # Create a mask for filtering based on the min and max values
    mask = (pos_l >= min_l) & (pos_l <= max_l) & (pos_w >= min_w) & (pos_w <= max_w)
    # Apply the mask to pos_l, pos_w, and deformation
    filtered_l = pos_l[mask]
    filtered_w = pos_w[mask]
    filtered_deformation = normal_def[mask]

    ### Plotting
    # Calculate the aspect ratio based on the range of x and y limits
    aspect_ratio = (max_l - min_l) / (max_w - min_w)

    # Assuming you want the width of the figure to be 10 inches
    width_inch = 10
    height_inch = width_inch / aspect_ratio  # Ensuring the l axis is longer than the w axis

    plt.figure(figsize=(width_inch, height_inch))
    sc = plt.scatter(filtered_l, filtered_w, c=filtered_deformation, cmap='seismic', s=10, vmin=-10, vmax=10)
    plt.colorbar(sc, label='Deformation Value')
    plt.xlabel('l', fontsize=14, weight='bold')
    plt.ylabel('w', fontsize=14, weight='bold')
    plt.title('Deformation in l-w Coordinate System', fontsize=16, weight='bold')

    # Set the aspect ratio to 'auto' to allow custom sizing
    plt.gca().set_aspect('auto')

    # Define the x and y limits
    plt.xlim(min_l, max_l)
    plt.ylim(min_w, max_w)

    # Draw the l-axis vector
    plt.arrow(min_l, 0, max_l-min_l, 0, fc='black', ec='black', linewidth=2, head_width=4, head_length=10, length_includes_head=True)

    # Draw the w-axis vectors
    plt.arrow(min_l, 0, 0, max_w, fc='black', ec='black', linewidth=2, head_width=4, head_length=10, length_includes_head=True)
    plt.arrow(min_l, 0, 0, min_w, fc='black', ec='black', linewidth=2, head_width=4, head_length=10, length_includes_head=True)

    # Enable grid
    plt.grid(False)

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

    
