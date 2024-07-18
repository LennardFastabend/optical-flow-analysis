import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Set the environment variable to avoid memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

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

#'''
### Cleft Segmentation ###
filtered_image = GaussFilter(image, kernel_size=(45,45), sigma=5)

edges = CannyEdgeDetection(filtered_image, threshold1=0, threshold2=45)

lines = HoughTransform(edges, threshold=125)

line_image = draw_line_image(image, lines, linecolor=(0,0,255))

clusters = cluster_lines(lines, num_clusters=2)

rep_lines = []
for cluster in clusters:
    rep_line = compute_representative_line(cluster)
    rep_lines.append(rep_line)

rep_line_image = draw_line_image(image, rep_lines, linecolor=(255,0,0))

intersection_point = compute_intersection(rep_lines)

cleft_mask = create_triangle_mask(image, rep_lines, intersection_point)

'''
# Display the original image, edges, and the image with lines
plt.figure(figsize=(15, 5))

plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(2, 3, 2)
plt.title('Filtered Image')
plt.imshow(filtered_image, cmap='gray')

plt.subplot(2, 3, 3)
plt.title('Canny Edges')
plt.imshow(edges, cmap='gray')

plt.subplot(2, 3, 4)
plt.title('Hough Lines')
plt.imshow(line_image)

plt.subplot(2, 3, 5)
plt.title('Mean Lines After Clustering')
plt.plot(intersection_point[0], intersection_point[1], 'ro',markersize = 3) 
plt.imshow(rep_line_image)

plt.subplot(2, 3, 6)
plt.title('Cleft Mask')
plt.imshow(cleft_mask, cmap='gray')

plt.show()
'''

### Growth Front Segmentation ###
# Apply the cleft mask to the image
image = cv2.bitwise_and(image, image, mask=cleft_mask)


kernel_size_sobel = 3
sobel_image = SobelFilter(image, kernel_size_sobel)

# Smooth image
kernel_size=(3,3)
sigma=1
filtered_image = GaussFilter(sobel_image, kernel_size, sigma)

check_point = (100,500)

threshold = 10
binary_mask = IntensitySegmentation(filtered_image, threshold)
binary_mask = invert_mask(binary_mask)
binary_mask = keep_contour_with_point(binary_mask, check_point)
contour_line = extract_contour_line(binary_mask)


plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Sobel + Gauss')
plt.imshow(filtered_image, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('Mask')
plt.imshow(binary_mask, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Image + Contour')
plt.imshow(image, cmap='gray')
plt.plot(contour_line[:, 0], contour_line[:, 1], marker='.', markersize=1, color='red', linestyle='-', linewidth=1)

plt.show()





'''
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
#plt.savefig(output_path / filename, dpi=600)   # save the figure to file
#plt.close()    # close the figure window

plt.show()
'''