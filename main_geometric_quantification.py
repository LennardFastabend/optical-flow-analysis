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

input_dir = Path(r'data\PhaseContrastCleft\P01\input\Aligned\LinearStackAlignmentSift_Gauss5px.avi')
#input_dir = Path(r'data\PhaseContrastCleft\P02\input\P08#39_live_W03-P02_aligned.avi')
#input_dir = Path(r'data\PhaseContrastCleft\P06\input\P08#39_live_W03-P06.avi') #Read in Aligned Data!

#input_dir = Path(r'data\PhaseContrastCleft\ContractilityComparisson\Blebbistatin\input\P08#39_live_W08-P01.avi')

output_dir = Path(r'data\PhaseContrastCleft\P01\temporal_average\images') 



input_reader = reader(root_dir, input_dir)
image_stack = input_reader.read_avi()

### Crop the image (resolves issues due to alignment of the images)
t, y, x = image_stack.shape
image_stack = image_stack[150:260, 100:y-100, 100:x-10] #crop the image and select the time
T_offset = 150



### calculate Example FlowFields for the defined time
dT=1
Tmax = 100
farneback_parameters = {"pyr_scale": 0.5,
                        "levels": 3,
                        "winsize": 25,
                        "iterations": 3,
                        "poly_n": 5, #10,
                        "poly_sigma": 1.2,
                        "flags": 0}

print('Start Farneback Analysis')
dt_OptFlow = 1
flowfield_stack = opflow.FlowFieldStack(image_stack, farneback_parameters, t0=0, tfin=Tmax-1, dt=dt_OptFlow)
print('Farneback Analysis Finished')
print(flowfield_stack.shape)
print()

#segmentation_generator = visualizer(root_dir, output_dir/Path('segmentation'))
#flowfield_generator = visualizer(root_dir, output_dir/Path('flowfields'))
#defmap_generator = visualizer(root_dir, output_dir/Path('defmap'))
geoquant_generator = visualizer(root_dir, output_dir/Path('ComponentGeoQuant'))
flowfield_generator = visualizer(root_dir, output_dir/Path('FlowField'))

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

max_shown_distance = 800
max_shown_displacement = 20
min_shown_displacement = -20

T0=0
step = 1

temp_scale = Tmax

df_parallel_list = []
df_normal_list = []
for T in np.arange(T0,T0+temp_scale,step):
    print(T)

    #if T >= 8: 
        #plt.imshow(image_stack[T,...])
        #plt.show()

    image = image_stack[T,...]
    meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
    defmap = opflow.calculateMagnitude(meanflowfield)

    # perform the segmentation
    cleft_mask, cleft_contour, front_mask, front_contour, representative_lines, intersection_point = Segmentation(image, segmentation_parameters)
    # define x and y coordinates of the cleft tip
    #intersec_x, intersec_y = intersection_point
    
    # find max x coordinate of growth front
    #xmax_front = np.argmax(front_contour[:, 0])
    #print('xmax of growth front:', xmax_front)

    # define tissue region based on masks
    tissue_mask = cv2.subtract(cleft_mask, front_mask)
    '''
    ### Example Vis of Mask and Masked Defmap
    plt.imshow(tissue_mask, cmap='gray')
    plt.plot(front_contour[:, 0], front_contour[:, 1], marker='.', markersize=0.2, color='red', linestyle='-', linewidth=1)
    plt.plot(cleft_contour[:, 0], cleft_contour[:, 1], marker='.', markersize=0.2, color='green', linestyle='-', linewidth=1)
    plt.show()

    masked_defmap = defmap * tissue_mask/255

    plt.imshow(masked_defmap, cmap='plasma', vmin=0, vmax=60)
    #plt.plot(front_contour[:, 0], front_contour[:, 1], marker='.', markersize=0.2, color='red', linestyle='-', linewidth=1)
    #plt.plot(cleft_contour[:, 0], cleft_contour[:, 1], marker='.', markersize=0.2, color='green', linestyle='-', linewidth=1)
    plt.colorbar()
    plt.show()
    sys.exit()
    #'''
    normal_vectors, distance_map = geoquant.ComputeNormalVectorField(tissue_mask, front_mask)

    FlowParallel, FlowNormal = geoquant.ComputeNormalAndParallelDisplacement(meanflowfield, normal_vectors)

    xmax_front = np.argmax(front_contour[:, 0])
    df_parallel = geoquant.GeometricQuantificationDistanceMap(FlowParallel, tissue_mask, distance_map, xmax_front, dx=100)
    df_normal = geoquant.GeometricQuantificationDistanceMap(FlowNormal, tissue_mask, distance_map, xmax_front, dx=100)
    ### Analysis of ROI along the central axis with width of +-dy pixels
    #df_parallel = geoquant.GeometricQuantificationDistanceMapROI(FlowParallel, tissue_mask, distance_map, intersection_point, dy=50)
    #df_normal = geoquant.GeometricQuantificationDistanceMapROI(FlowNormal, tissue_mask, distance_map, intersection_point, dy=50)
    #'''
    ###Visualisation of Normal/parallel component of mean flow field as scatter plots

    title_parallel = 'Displacement Parallel to the Growth Front at Time '+ str(T+T_offset) + ' with $dt_{OptFlow}$=' + str(dt_OptFlow)
    filename_parallel = 'geoquant_parallel' + str(T+T_offset)
    geoquant_generator.saveGeometricQuantificationScatterPlot(df_parallel, max_shown_distance, min_shown_displacement, max_shown_displacement, title_parallel, filename_parallel, c='green')
    title_normal = 'Displacement Normal to the Growth Front at Time: '+ str(T+T_offset) + ' with $dt_{OptFlow}$=' + str(dt_OptFlow)
    filename_normal = 'geoquant_normal' + str(T+T_offset)
    geoquant_generator.saveGeometricQuantificationScatterPlot(df_normal, max_shown_distance, min_shown_displacement, max_shown_displacement, title_normal, filename_normal, c='red')

    flowfield_generator.saveFlowField(image, meanflowfield, title='FlowField at Time: '+str(T+T_offset)+'-'+str(T+T_offset+dt_OptFlow), filename='FlowField'+str(T+T_offset), step=15, epsilon=0, scale=0.5)
    #'''

    '''
    ###Visualisation of Normal/parallel component of mean flow field as binned statistics
    bin_size = 20

    title_parallel = 'Displacement Parallel to the Growth Front at Time '+ str(T+T_offset) + ' with $dt_{OptFlow}$=' + str(dt_OptFlow)
    filename_parallel = 'geoquant_parallel' + str(T+T_offset)
    geoquant_generator.saveGeometricQuantificationBinnedStatistics(df_parallel, bin_size, max_shown_distance, min_shown_displacement, max_shown_displacement, title_parallel, filename_parallel)
    title_normal = 'Displacement Normal to the Growth Front at Time: '+ str(T+T_offset) + ' with $dt_{OptFlow}$=' + str(dt_OptFlow)
    filename_normal = 'geoquant_normal' + str(T+T_offset)
    geoquant_generator.saveGeometricQuantificationBinnedStatistics(df_normal, bin_size, max_shown_distance, min_shown_displacement, max_shown_displacement, title_normal, filename_normal)
    #'''


    df_parallel_list.append(df_parallel)
    df_normal_list.append(df_normal)

#'''
### Visualisation of Cumulative Normal/Parallel component as scatter heat map (dT and step have to be 1!!!)

title_parallel = 'Displacement Parallel to the Growth Front at Time: '+ str(T0+T_offset) + '-' + str(T0+temp_scale+T_offset)
filename_parallel = 'CumulativeGeoquant_parallel' + str(T0+T_offset) + '-' + str(T0+temp_scale+T_offset)
#geoquant_generator.saveCumulativeGeometricQuantificationScatterPlot(df_parallel_list, max_shown_distance, min_shown_displacement, max_shown_displacement, title_parallel, filename_parallel, c='green') #Note: Set dT = 1 for this!!!
geoquant_generator.saveCumulativeGeometricQuantificationHeatMap(df_parallel_list, max_shown_distance, min_shown_displacement, max_shown_displacement, title_parallel, filename_parallel+str('Heatmap'))

title_normal = 'Displacement Normal to the Growth Front at Time: '+ str(T0+T_offset) + '-' + str(T0+temp_scale+T_offset)
filename_normal = 'CumulativeGeoquant_normal' + str(T0+T_offset) + '-' + str(T0+temp_scale+T_offset)
#geoquant_generator.saveCumulativeGeometricQuantificationScatterPlot(df_normal_list, max_shown_distance, min_shown_displacement, max_shown_displacement, title_normal, filename_normal, c='red') #Note: Set dT = 1 for this!!!
geoquant_generator.saveCumulativeGeometricQuantificationHeatMap(df_normal_list, max_shown_distance, min_shown_displacement, max_shown_displacement, title_normal, filename_normal+str('Heatmap'))

#segmentation_generator.saveSegmentationMasks(image, front_contour, cleft_contour, title='Segmentation at Time:'+str(T), filename='segmentation'+str(T))
#flowfield_generator.saveFlowField(image, meanflowfield, title='FlowField at Time: '+str(T)+'-'+str(T+dT), filename='FlowField'+str(T), step=20, epsilon=0)
#'''

#'''
### Visualisation of Cumulative Normal/Parallel component as binned statistics (dT and step have to be 1!!!)

bin_size = 20
smoothing_winsize = 100

title_parallel = 'Displacement Parallel to the Growth Front at Time: '+ str(T0+T_offset) + '-' + str(T0+temp_scale+T_offset)
filename_parallel = 'CumulativeGeoquant_parallel' + str(T0+T_offset) + '-' + str(T0+temp_scale+T_offset)
geoquant_generator.saveGeometricQuantificationCumulativeBinnedStatistics(df_parallel_list, bin_size, max_shown_distance, min_shown_displacement, max_shown_displacement, title_parallel, filename_parallel)
geoquant_generator.saveGeometricQuantificationCumulativePercentileBands(df_parallel_list, smoothing_winsize, max_shown_distance, min_shown_displacement, max_shown_displacement, title_parallel, filename_parallel+str('PercentileBands'))

title_normal = 'Displacement Normal to the Growth Front at Time: '+ str(T0+T_offset) + '-' + str(T0+temp_scale+T_offset)
filename_normal = 'CumulativeGeoquant_normal' + str(T0+T_offset) + '-' + str(T0+temp_scale+T_offset)
geoquant_generator.saveGeometricQuantificationCumulativeBinnedStatistics(df_normal_list, bin_size, max_shown_distance, min_shown_displacement, max_shown_displacement, title_normal, filename_normal)
geoquant_generator.saveGeometricQuantificationCumulativePercentileBands(df_normal_list, smoothing_winsize, max_shown_distance, min_shown_displacement, max_shown_displacement, title_normal, filename_normal+str('PercentileBands'))
#'''



sys.exit()





'''
title_parallel = 'Displacement Parallel to the Growth Frant at Time: '+ str(T) + '-' + str(T+dT)
filename_parallel = 'geoquant_parallel' + str(T)
geoquant_generator.saveGeometricQuantificationScatterPlot(df_parallel, max_shown_distance, max_shown_displacement, title_parallel, filename_parallel, c='green')
title_normal = 'Displacement Normal to the Growth Frant at Time: '+ str(T) + '-' + str(T+dT)
filename_normal = 'geoquant_normal' + str(T)
geoquant_generator.saveGeometricQuantificationScatterPlot(df_normal, max_shown_distance, max_shown_displacement, title_normal, filename_normal, c='red')
'''
'''
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Displacement Normal to Growth Front')
plt.imshow(FlowNormal, cmap='seismic', vmin=-10, vmax=10)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title('Displacement Parallel to Growth Front')
plt.imshow(FlowParallel, cmap='seismic', vmin=-10, vmax=10)
plt.colorbar()

output_path = output_dir / Path('ComponentAnalysis')
filename = 'ComponentAnalysis' + str(T)
plt.savefig(output_path / filename, dpi=600)   # save the figure to file
plt.close()    # close the figure window
'''







max_shown_distance = 1000
max_shown_displacement = 15
'''
Example defmap x/y-component analysis in scatter plots
'''
T0=0
step = 25
for T in np.arange(T0,Tmax-step,step):
    print(T)

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
    df = geoquant.DirectionalGeometricQuantification(meanflowfield, tissue_mask, front_contour, xmax_front, dx=100)

    segmentation_generator.saveSegmentationMasks(image, front_contour, cleft_contour, title='Segmentation at Time:'+str(T), filename='segmentation'+str(T))
    flowfield_generator.saveFlowField(image_stack[T,...], meanflowfield, title='FlowField at Time: '+str(T)+'-'+str(T+dT), filename='FlowField'+str(T), step=15, epsilon=0)

    title = 'Displacement vs. Distance to Growth Front at Time: '+ str(T) + '-' + str(T+dT)
    filename = 'geoquant' + str(T)
    geoquant_generator.saveDirectionalGeometricQuantificationScatterPlot(df, max_shown_distance, max_shown_displacement, title, filename, displacement_type='x')
    geoquant_generator.saveDirectionalGeometricQuantificationScatterPlot(df, max_shown_distance, max_shown_displacement, title, filename, displacement_type='y')









sys.exit()
'''
Example defmap magnitude analysis in scatter plots and cumulative scatter plots
'''
# paramters for geoquant visualisation
bin_size = 20
max_shown_distance = 1000
max_shown_displacement = 15

df_list = []
T0=150
temp_scale = 100
for T in np.arange(T0,T0+temp_scale,dT):
    print(T)

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
    df_list.append(df)

    #title = 'Displacement vs. Distance to Growth Front at Time: '+ str(T) + '-' + str(T+dT)
    #filename = 'geoquant' + str(T)
    #geoquant_generator.saveGeometricQuantificationBinnedStatistics(df, bin_size, max_shown_distance, max_shown_displacement, title, filename+'binned')
    #geoquant_generator.saveGeometricQuantificationScatterPlot(df, max_shown_distance, max_shown_displacement, title, filename)
    #defmap_generator.saveDeformationMap(defmap, min=0, max=10, title='DefMap at Time: '+str(T)+'-'+str(T+dT), filename='DefMap'+str(T))
    #segmentation_generator.saveSegmentationMasks(image, front_contour, cleft_contour, title='Segmentation at Time:'+str(T), filename='segmentation'+str(T))
    #flowfield_generator.saveFlowField(image_stack[T,...], meanflowfield, title='FlowField at Time: '+str(T)+'-'+str(T+dT), filename='FlowField'+str(T), step=20, epsilon=0)

title = 'Displacement vs. Distance to Growth Front at Time: '+ str(T0) + '-' + str(T0+temp_scale)
filename = 'CumulativeGeoquant' + str(T0) + '-' + str(T0+temp_scale)
geoquant_generator.saveCumulativeGeometricQuantificationScatterPlot(df_list, max_shown_distance, max_shown_displacement, title, filename) #Note: Set dT = 1 for this!!!
filename = 'CumulativeGeoquantHeatmap' + str(T0) + '-' + str(T0+temp_scale)
geoquant_generator.saveCumulativeGeometricQuantificationHeatMap(df_list, max_shown_distance, max_shown_displacement, title, filename, bins=1000, cmap='jet')

























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