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
image_stack = image_stack[0:50, 100:y-100, 50:x-30] #crop the image
T_offset = 0

### calculate Example FlowFields for the defined time
dT=1
Tmax = 10
farneback_parameters = {"pyr_scale": 0.5,
                        "levels": 3,
                        "winsize": 5,#15,
                        "iterations": 3,
                        "poly_n": 5,
                        "poly_sigma": 1.2,
                        "flags": 0}
#'''
print('Start Farneback Analysis')
dt_OptFlow = 10
flowfield_stack = opflow.FlowFieldStack(image_stack, farneback_parameters, t0=0, tfin=Tmax, dt=dt_OptFlow)
print(flowfield_stack.shape)
print('Farneback Analysis Finished')
print()
#'''

segmentation_generator = visualizer(root_dir, output_dir/Path('segmentation'))
cleft_edge_generator = visualizer(root_dir, output_dir/Path('cleft_edge_flow'))
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

T0=0
step = 1
temp_scale = Tmax

mean_normal_def = np.zeros(image_stack[0,...].shape)
mean_parallel_def = np.zeros(image_stack[0,...].shape)

for T in np.arange(T0,T0+temp_scale,step):
    print(T)

    image = image_stack[T,...]
    meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
    defmap = opflow.calculateMagnitude(meanflowfield)

    # perform the segmentation
    #cleft_mask, cleft_contour, front_mask, front_contour, edge_lines = Segmentation(image, segmentation_parameters)
    cleft_mask, cleft_contour, front_mask, front_contour, edge_lines, cleft_tip = Segmentation(image, segmentation_parameters)

    line_intersection = cleft_tip #compute_intersection(edge_lines)
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

    mean_normal_def += normal_def/temp_scale
    mean_parallel_def += parallel_def/temp_scale



    ### Visualize pos_l, pos_w and deformation values in a l-w-plot

    #define max/min values for the l and w position (this defines the visualized ROI)
    min_l, max_l = 100,1300 #length range in pixels from the origin
    min_w, max_w = -150,150 #width range in pixels from the origin (positive is inside the cleft, negative outside)

    filename = 'LowerCleftEdge'+ str(T)
    cleft_edge_generator.saveCleftEdgeFlow(pos_l, pos_w, normal_def, parallel_def, filename, min_l, max_l, min_w, max_w, min_def=-20, max_def=20)

#filename = 'MeanLowerCleftEdge'+ str(T0+T_offset) + '-' + str(T0+temp_scale+T_offset)
#cleft_edge_generator.saveCleftEdgeFlow(pos_l, pos_w, mean_normal_def, mean_parallel_def, filename, min_l, max_l, min_w, max_w, min_def=-5, max_def=5)



'''
# Create a mask for filtering based on the min and max values
mask = (pos_l >= min_l) & (pos_l <= max_l) & (pos_w >= min_w) & (pos_w <= max_w)
# Apply the mask to pos_l, pos_w, and deformation
filtered_l = pos_l[mask]
filtered_w = pos_w[mask]
filtered_deformation_normal = normal_def[mask]
filtered_deformation_parallel = parallel_def[mask]

### Plotting
# Calculate the aspect ratio based on the range of x and y limits
aspect_ratio = (max_l - min_l) / (max_w - min_w)

# Assuming you want the width of the figure to be 10 inches
width_inch = 10
height_inch = width_inch / aspect_ratio  # Ensuring the l axis is longer than the w axis

# Create a figure with two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width_inch, height_inch * 2))  # height_inch * 2 for stacking subplots

# Plot in the first subplot
sc1 = ax1.scatter(filtered_l, filtered_w, c=filtered_deformation_normal, cmap='seismic', s=10, vmin=-10, vmax=10)
ax1.set_xlabel('l', fontsize=14)
ax1.set_ylabel('w', fontsize=14)
ax1.set_title('Normal Deformation in l-w Coordinate System', fontsize=16)
ax1.set_xlim(min_l, max_l)
ax1.set_ylim(min_w, max_w)
ax1.plot([min_l, max_l], [0, 0], color='black', linewidth=2)
ax1.plot([min_l, min_l], [min_w, max_w], color='black', linewidth=2)
ax1.grid(True)
fig.colorbar(sc1, ax=ax1, label='Normal Deformation')

# Plot in the second subplot
sc2 = ax2.scatter(filtered_l, filtered_w, c=filtered_deformation_parallel, cmap='seismic', s=10, vmin=-10, vmax=10)
ax2.set_xlabel('l', fontsize=14)
ax2.set_ylabel('w', fontsize=14)
ax2.set_title('Parallel Deformation in l-w Coordinate System', fontsize=16)
ax2.set_xlim(min_l, max_l)
ax2.set_ylim(min_w, max_w)
ax2.plot([min_l, max_l], [0, 0], color='black', linewidth=2)
ax2.plot([min_l, min_l], [min_w, max_w], color='black', linewidth=2)
ax2.grid(True)
fig.colorbar(sc2, ax=ax2, label='Parallel Deformation')

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()
'''







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

    
