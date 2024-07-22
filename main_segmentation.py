import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Set the environment variable to avoid memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

from data_segmentation.segmentation import Segmentation
import data_reader.reader as reader
from output_generation.visualizer import visualizer



root_dir = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis') #path to repository
input_dir = Path(r'data\PhaseContrastCleft\P01\input\P08#39_live_W03-P01.avi')
output_path = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis\data\PhaseContrastCleft\P01\segmentation_testing')

image_reader = reader.reader(root_dir, input_dir)
image_stack = image_reader.read_avi()

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
#'''
for t in np.arange(262,350,1):
#t = 263
    image = image_stack[t,...]
    cleft_mask, cleft_contour, front_mask, front_contour = Segmentation(image, segmentation_parameters)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title('Image + Contours')
    plt.imshow(image, cmap='gray')
    plt.plot(front_contour[:, 0], front_contour[:, 1], marker='.', markersize=0.2, color='red', linestyle='-', linewidth=0.5)
    plt.plot(cleft_contour[:, 0], cleft_contour[:, 1], marker='.', markersize=0.2, color='green', linestyle='-', linewidth=0.5)

    plt.subplot(1, 3, 2)
    plt.title('Cleft Mask')
    plt.imshow(cleft_mask, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Growth Front Mask')
    plt.imshow(front_mask, cmap='gray')

    filename = 'Segmentation'+str(t)
    plt.savefig(output_path / filename, dpi=600)   # save the figure to file
    plt.close()    # close the figure window
    #plt.show()




























'''
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
    cleft_contour = extract_contour_line(cleft_mask)

    
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
    

    ### Growth Front Segmentation ###
    # Apply the cleft mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=cleft_mask)

    kernel_size_sobel = 3
    sobel_image = SobelFilter(masked_image, kernel_size_sobel)

    # Smooth image
    kernel_size=(3,3)
    sigma=1
    filtered_image = GaussFilter(sobel_image, kernel_size, sigma)

    threshold = 10
    front_mask = IntensitySegmentation(filtered_image, threshold)
    front_mask = invert_mask(front_mask)
    check_point = (0,intersection_point[1]) #based on the cleft edges (only works, when the cleft opens to the left)
    front_mask = keep_contour_with_point(front_mask, check_point)


    # Smooth Mask Contour
    # Apply morphological operations to remove small noise
    kernel = np.ones((25, 25), np.uint8)
    front_mask = cv2.morphologyEx(front_mask, cv2.MORPH_OPEN, kernel)

    ### Erode the mask (includes inversion to erode in the right direction)
    front_mask = invert_mask(front_mask)
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    front_mask = cv2.erode(front_mask, kernel, iterations=3)
    front_mask = invert_mask(front_mask)
    front_mask = keep_largest_edge_contour(front_mask) #this catches errors due to multiple masked regions after mask-smoothing!

    ### get the front contour
    front_contour = extract_contour_line(front_mask)
'''