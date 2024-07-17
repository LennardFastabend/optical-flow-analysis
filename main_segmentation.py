import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from data_segmentation.segmentation import *
import data_reader.reader as reader
from output_generation.visualizer import visualizer

root_dir = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis') #path to repository
input_dir = Path(r'data\PhaseContrastCleft\P01\input\P08#39_live_W03-P01.avi')
output_path = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis\data\PhaseContrastCleft\P01\segmentation_testing')

image_reader = reader.reader(root_dir, input_dir)
image_stack = image_reader.read_avi()
#'''
#for t in np.arange(0,350,5):
t = 150
image = image_stack[t,...]
front_mask,filtered_image_front,front_contour_line = SegementGrowthFront(image)
cleft_mask,filtered_image_cleft,cleft_contour_line = SegementCleft(image)

#Visualise masks and image
'''
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
#plt.savefig(output_path / filename, dpi=600)   # save the figure to file
#plt.close()    # close the figure window

plt.show()
'''

'''
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Filtered Image')
plt.imshow(filtered_image_front, cmap='gray')
'''
'''
#plt.subplot(1, 2, 2)
plt.title('Original Image + Contour')
plt.imshow(image, cmap='gray')
plt.plot(front_contour_line[:, 0], front_contour_line[:, 1], marker='.', markersize=1, color='red', linestyle='-', linewidth=1)

filename = '4_FrontSeg'+str(t)
plt.savefig(output_path / filename, dpi=600)   # save the figure to file
plt.close()    # close the figure window
'''

#################################################
'''
edges = CannyEdgeDetection(image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Filtered Image')
plt.imshow(edges, cmap='gray')

plt.show()
'''

# Apply the Canny edge detection 
edges = CannyEdgeDetection(image)

# Apply Hough Line Transform
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=150)

    # Check if any lines were detected
if lines is not None:
    # Convert lines to a more understandable format and sort by votes
    lines = sorted(lines, key=lambda line: line[0][0], reverse=True)  # Sorting by rho value

    # Select the top two lines
    top_lines = lines[:2]

    # Create a copy of the original image to draw lines on
    line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for line in top_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the original image, edges, and the image with lines
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title('Canny Edges')
    plt.imshow(edges, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title('Top 2 Hough Lines')
    plt.imshow(line_image)
    
    plt.show()
else:
    print("No lines were detected.")