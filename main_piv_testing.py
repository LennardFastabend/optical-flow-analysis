from data_reader.reader import reader
from output_generation.visualizer import visualizer
import data_analysis.particle_image_velocimetry as piv
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
output_dir = Path(r'data\PhaseContrastCleft\P01\PIV_Testing\GeoQuantT150_dt1')

input_reader = reader(root_dir, input_dir)
image_stack = input_reader.read_avi()


### Crop the image (resolves issues due to alignment of the images)
t, y, x = image_stack.shape
image_stack = image_stack[150:260, 100:y-100, 100:x-10] #crop the image and select the time

T_offset = 150
Tmax = 100

image_generator = visualizer(root_dir, output_dir)


max_shown_distance = 800
max_shown_displacement = 10
min_shown_displacement = -10
max_magnitude = 10

winsize = 20
dt_PIV=1
piv_parameters =  {
    "window_size": winsize,              # Size of interrogation window (32x32 pixels)
    "overlap": winsize-5,                  # Overlap between adjacent windows (16 pixels)
}
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

print('Start PIV Analysis')
piv_stack = piv.PIVStack(image_stack, piv_parameters, t0=0, tfin=Tmax, dt=dt_PIV)
print(piv_stack.shape)
print('PIV Analysis Finished')
print()

df_parallel_list = []
df_normal_list = []

for T in np.arange(T0,T0+temp_scale,step):
    image = image_stack[T,...]

    # perform the segmentation
    cleft_mask, cleft_contour, front_mask, front_contour, representative_lines, intersection_point = Segmentation(image, segmentation_parameters)
    # define tissue region based on masks
    tissue_mask = cv2.subtract(cleft_mask, front_mask)

    flowfield = piv_stack[T,...]
    defmap = opflow.calculateMagnitude(flowfield)

    normal_vectors, distance_map = geoquant.ComputeNormalVectorField(tissue_mask, front_mask)

    FlowParallel, FlowNormal = geoquant.ComputeNormalAndParallelDisplacement(flowfield, normal_vectors)

    xmax_front = np.argmax(front_contour[:, 0])
    df_parallel = geoquant.GeometricQuantificationDistanceMap(FlowParallel, tissue_mask, distance_map, xmax_front, dx=100)
    df_normal = geoquant.GeometricQuantificationDistanceMap(FlowNormal, tissue_mask, distance_map, xmax_front, dx=100)



    # Save initial and final Image
    #image_generator.saveImage(image_stack[0,...], title='Cleft at Time: '+str(T_offset), filename='Image'+str(T_offset))
    #image_generator.saveImage(image_stack[0+dt_PIV,...], title='Cleft at Time: '+str(T_offset+dt_PIV), filename='Image'+str(T_offset+dt_PIV))
                                
    # Save FlowField
    #image_generator.saveFlowField(image_stack[0,...], flowfield, title='FlowField at Time: '+str(T_offset)+'-'+str(T_offset+dt_PIV), filename='FlowField'+str(T_offset)+'winsize'+str(winsize), step=1, epsilon=0, scale=0.5)
    # Save DefMap
    #image_generator.saveDeformationMap(defmap, min=0, max=max_magnitude, title='DefMap at Time: '+str(T_offset)+'-'+str(T_offset+dt_PIV), filename='DefMap'+str(T_offset))

    df_parallel_list.append(df_parallel)
    df_normal_list.append(df_normal)

### Visualisation of Cumulative Normal/Parallel component as scatter heat map (dT and step have to be 1!!!)

title_parallel = 'Displacement Parallel to the Growth Front at Time: '+ str(T0+T_offset) + '-' + str(T0+temp_scale+T_offset)
filename_parallel = 'CumulativeGeoquant_parallel' + str(T0+T_offset) + '-' + str(T0+temp_scale+T_offset)
image_generator.saveCumulativeGeometricQuantificationScatterPlot(df_parallel_list, max_shown_distance, min_shown_displacement, max_shown_displacement, title_parallel, filename_parallel, c='green') #Note: Set dT = 1 for this!!!
image_generator.saveCumulativeGeometricQuantificationHeatMap(df_parallel_list, max_shown_distance, min_shown_displacement, max_shown_displacement, title_parallel, filename_parallel+str('Heatmap'))

title_normal = 'Displacement Normal to the Growth Front at Time: '+ str(T0+T_offset) + '-' + str(T0+temp_scale+T_offset)
filename_normal = 'CumulativeGeoquant_normal' + str(T0+T_offset) + '-' + str(T0+temp_scale+T_offset)
image_generator.saveCumulativeGeometricQuantificationScatterPlot(df_normal_list, max_shown_distance, min_shown_displacement, max_shown_displacement, title_normal, filename_normal, c='red') #Note: Set dT = 1 for this!!!
image_generator.saveCumulativeGeometricQuantificationHeatMap(df_normal_list, max_shown_distance, min_shown_displacement, max_shown_displacement, title_normal, filename_normal+str('Heatmap'))