import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from data_segmentation.segmentation import *
import data_reader.reader as reader
from output_generation.visualizer import visualizer

root_dir = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis') #path to repository
input_dir = Path(r'data\PhaseContrastCleft\P01\input\Aligned\LinearStackAlignmentSift_Gauss5px.avi')
output_path = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis\data\PhaseContrastCleft\P01\segmentation')

image_reader = reader.reader(root_dir, input_dir)
image_stack = image_reader.read_avi()
#'''
for t in np.arange(0,350,5):
    image = image_stack[t,...]
    front_mask,filtered_image_front,front_contour_line = SegementGrowthFront(image)
    cleft_mask,filtered_image_cleft,cleft_contour_line = SegementCleft(image)

    #Visualise masks and image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image + Contours')
    plt.imshow(image, cmap='gray')
    plt.plot(front_contour_line[:, 0], front_contour_line[:, 1], marker='.', markersize=1, color='red', linestyle='-', linewidth=1)
    plt.plot(cleft_contour_line[:, 0], cleft_contour_line[:, 1], marker='.', markersize=1, color='orange', linestyle='-', linewidth=1)

    plt.subplot(1, 3, 2)
    plt.title('Growth Front Mask')
    plt.imshow(front_mask, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Cleft Mask')
    plt.imshow(cleft_mask, cmap='gray')

    filename = 'Segmentation'+str(t)
    plt.savefig(output_path / filename, dpi=600)   # save the figure to file
    plt.close()    # close the figure window

    plt.show()
#'''