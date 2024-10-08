import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.ndimage import label

# Set the environment variable to avoid memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"


from skimage import feature
from sklearn.cluster import KMeans



def IntensitySegmentation(image, threshold_value):
     # Convert the image to grayscale if it is not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    _, binary_mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_mask

def invert_mask(binary_mask):
    return 255-binary_mask

def keep_largest_edge_contour(binary_mask):
    # Find all contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return the original mask
    if not contours:
        return binary_mask

    # Get image dimensions
    height, width = binary_mask.shape

    # Initialize variables to keep track of the largest edge-connected contour
    max_area = 0
    largest_contour = None

    # Define edge coordinates
    edge_mask = np.zeros_like(binary_mask)
    edge_mask[0, :] = 255
    edge_mask[-1, :] = 255
    edge_mask[:, 0] = 255
    edge_mask[:, -1] = 255

    # Iterate through each contour to find edge-connected contours
    for contour in contours:
        # Create a mask for the current contour
        contour_mask = np.zeros_like(binary_mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Check if the contour is connected to the edge
        if np.any(cv2.bitwise_and(contour_mask, edge_mask)):
            # Calculate the area of the contour
            contour_area = cv2.contourArea(contour)

            # Update the largest contour if this one is larger
            if contour_area > max_area:
                max_area = contour_area
                largest_contour = contour

    # Create a mask for the largest edge-connected contour
    largest_contour_mask = np.zeros_like(binary_mask)
    if largest_contour is not None:
        cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return largest_contour_mask

def keep_contour_with_point(binary_mask, point):
    # Find all contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return the original mask
    if not contours:
        return binary_mask

    # Initialize variable to keep track of the contour containing the point
    containing_contour = None

    # Iterate through each contour to find the one containing the point
    for contour in contours:
        if cv2.pointPolygonTest(contour, point, False) >= 0:
            containing_contour = contour
            break

    # Create a mask for the contour containing the point
    containing_contour_mask = np.zeros_like(binary_mask)
    if containing_contour is not None:
        cv2.drawContours(containing_contour_mask, [containing_contour], -1, 255, thickness=cv2.FILLED)

    return containing_contour_mask

def remove_contour_with_point(binary_mask, point):
    # Find all contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return the original mask
    if not contours:
        return binary_mask

    # Initialize list to keep track of the contours to be kept
    contours_to_keep = []

    # Iterate through each contour to find the one containing the point
    for contour in contours:
        if cv2.pointPolygonTest(contour, point, False) < 0:
            contours_to_keep.append(contour)

    # Create a mask for the contours to be kept
    result_mask = np.zeros_like(binary_mask)
    if contours_to_keep:
        cv2.drawContours(result_mask, contours_to_keep, -1, 255, thickness=cv2.FILLED)

    return result_mask


def extract_contour_line(binary_mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Ensure at least one contour was found
    if contours:
        # Assume the largest contour (first one) is the one we're interested in
        contour = contours[0]

        # Optionally, approximate the contour to reduce points
        epsilon = 0.0001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Extract the contour line
        contour_line = approx.squeeze().tolist()
        contour_line = np.array(contour_line)
        return contour_line
    else:
        return None

##########################################################################
def GaussFilter(image, kernel_size, sigma):
    kernel_size = (kernel_size,kernel_size)
    filtered_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return filtered_image

def SobelFilter(image, kernel_size):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_edges = cv2.magnitude(sobelx, sobely)
    # Normalize to 8-bit scale
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX)
    sobel_edges = np.uint8(sobel_edges)
    return sobel_edges


###################
### Cleft Edges ###
###################
def CannyEdgeDetection(image, threshold1=0, threshold2=45):
    # Apply the Canny edge detection filter
    edges = cv2.Canny(image, threshold1=threshold1, threshold2=threshold2)
    #edges = feature.canny(image, sigma = 5)

    # Convert boolean array to uint8
    #edges = (edges * 255).astype(np.uint8)

    return edges

def HoughTransform(edges, threshold=125):
    # Apply Hough Line Transform
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/(180*2), threshold=threshold)
    # Convert lines to a more understandable format and sort by votes
    lines = sorted(lines, key=lambda line: line[0][0], reverse=True)  # Sorting by rho value
    # Select the top lines
    #lines = lines[:5] # this makes it worse
    return lines

def cluster_lines(lines, num_clusters=2):
    # Extract (rho, theta) parameters from lines
    rho_theta = np.array([[line[0][0], line[0][1]] for line in lines])

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(rho_theta)
    labels = kmeans.labels_

    # Separate the lines into clusters
    clusters = [[] for _ in range(num_clusters)]
    for label, line in zip(labels, lines):
        clusters[label].append(line)

    return clusters

def compute_representative_line(cluster):
    # Compute average rho and theta for the cluster
    rho_avg = np.mean([line[0][0] for line in cluster])
    theta_avg = np.mean([line[0][1] for line in cluster])
    return [[rho_avg, theta_avg]]  # Return in the same format as HoughLines output

def draw_line_image(image, lines, linecolor=(0,0,255)):
    # Create a copy of the original image to draw lines on
    line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))
        cv2.line(line_image, (x1, y1), (x2, y2), linecolor, 2)
    return line_image

def compute_intersection(lines):
    # Extract rho and theta for both lines
    rho1, theta1 = lines[0][0]
    rho2, theta2 = lines[1][0]

    # Convert polar coordinates to Cartesian line equation coefficients
    A1 = np.cos(theta1)
    B1 = np.sin(theta1)
    C1 = rho1

    A2 = np.cos(theta2)
    B2 = np.sin(theta2)
    C2 = rho2

    # Solve the system of linear equations to find the intersection point
    A = np.array([[A1, B1], [A2, B2]])
    C = np.array([C1, C2])

    # Check if lines are parallel (det(A) == 0)
    if np.linalg.det(A) == 0:
        return None  # Lines are parallel or coincident

    # Calculate intersection point
    intersection = np.linalg.solve(A, C)
    
    return intersection

def compute_left_border_intersection(line, image):
    # Get the dimensions of the image
    image_height, image_width = image.shape  # shape returns (height, width)

    # Extract rho and theta for the line
    rho, theta = line[0]

    # Convert polar coordinates to Cartesian line equation coefficients
    A = np.cos(theta)
    B = np.sin(theta)
    C = rho

    # The equation of the line in Cartesian form is: Ax + By = C
    # The left border of the image corresponds to x = 0.
    
    # For x = 0, the equation simplifies to: B * y = C
    # So, y = C / B
    
    if B == 0:  # This would imply that the line is horizontal and does not intersect the left border
        return None
    
    y = C / B

    # Check if the intersection is within the bounds of the image height
    if 0 <= y <= image_height:
        intersection = [0, int(y)]  # The intersection point at the left border (x=0)
        return intersection
    else:
        return None  # Intersection is out of image bounds

def create_triangle_mask(image, lines, intersection):
    # Image dimensions
    height, width = image.shape

    # Points on the lines
    rho1, theta1 = lines[0][0]
    rho2, theta2 = lines[1][0]

    # Compute two points on each line far enough to cover the image
    x0_1 = np.cos(theta1) * rho1
    y0_1 = np.sin(theta1) * rho1
    point1_line1 = (int(x0_1 + 1000 * (-np.sin(theta1))), int(y0_1 + 1000 * (np.cos(theta1))))
    point2_line1 = (int(x0_1 - 1000 * (-np.sin(theta1))), int(y0_1 - 1000 * (np.cos(theta1))))

    x0_2 = np.cos(theta2) * rho2
    y0_2 = np.sin(theta2) * rho2
    point1_line2 = (int(x0_2 + 1000 * (-np.sin(theta2))), int(y0_2 + 1000 * (np.cos(theta2))))
    point2_line2 = (int(x0_2 - 1000 * (-np.sin(theta2))), int(y0_2 - 1000 * (np.cos(theta2))))

    # Create a mask with zeros (black)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Define triangle vertices
    triangle_vertices = np.array([[
        (int(intersection[0]), int(intersection[1])),  # Intersection point
        point1_line1,
        point1_line2
    ]], dtype=np.int32)

    # Fill the triangle on the mask with white (255)
    cv2.fillPoly(mask, triangle_vertices, 255)

    return mask


############################################################################
def smooth_contour(contour_points, sigma=2):
    """
    Smooths the contour points using Gaussian smoothing.
    
    Parameters:
    - contour_points (np.ndarray): The input contour points as a Nx2 array.
    - sigma (float): The standard deviation for Gaussian kernel.
    
    Returns:
    - smoothed_contour (np.ndarray): The smoothed contour points as a Nx2 array.
    """
    # Discard points with an x value of 0
    contour_points = contour_points[contour_points[:, 0] != 0]
    
    # Extract x and y coordinates
    x = contour_points[:, 0]
    y = contour_points[:, 1]
    
    # Apply Gaussian smoothing
    smoothed_x = scipy.ndimage.gaussian_filter1d(x, sigma=sigma)
    smoothed_y = scipy.ndimage.gaussian_filter1d(y, sigma=sigma)
    
    # Combine the smoothed x and y coordinates
    smoothed_contour = np.vstack((smoothed_x, smoothed_y)).T
    
    return smoothed_contour

############################################################################

def Segmentation(image, segpar):
    '''
    This Segmentation takes an input image + segmantation_patrameter_dictionary, 
    to create masks and contour lines of the cleft edges and growth front

    Note: The Segmentation only works if the cleft opens to the right side of the image:
    Otherwise the definition of the 'check_point' var has to be changed, or the image has to be flipped!
    '''
    ### Cleft Segmentation ###
    filtered_image = GaussFilter(image, kernel_size=segpar["cleft_gauss_ksize"], sigma=segpar["cleft_gauss_sigma"])

    edges = CannyEdgeDetection(filtered_image, threshold1=segpar["cleft_canny_th1"], threshold2=segpar["cleft_canny_th2"])

    lines = HoughTransform(edges, threshold=segpar["cleft_hough_th"])

    #line_image = draw_line_image(image, lines, linecolor=(0,0,255))

    clusters = cluster_lines(lines, num_clusters=2)

    rep_lines = []
    for cluster in clusters:
        rep_line = compute_representative_line(cluster)
        rep_lines.append(rep_line)

    #rep_line_image = draw_line_image(image, rep_lines, linecolor=(255,0,0))

    intersection_point = compute_intersection(rep_lines)

    cleft_mask = create_triangle_mask(image, rep_lines, intersection_point)
    cleft_contour = extract_contour_line(cleft_mask)


    ### Growth Front Segmentation ###
    masked_image = cv2.bitwise_and(image, image, mask=cleft_mask)

    sobel_image = SobelFilter(masked_image, kernel_size=segpar["front_sobel_ksize"])

    # Smooth image
    #kernel_size=(3,3)
    #sigma=1
    filtered_image = GaussFilter(sobel_image, kernel_size=segpar["front_gauss_ksize"], sigma=segpar["front_gauss_sigma"])



    threshold = segpar["front_segmentation_th"]
    front_mask = IntensitySegmentation(filtered_image, threshold)
    front_mask = invert_mask(front_mask)
    check_point = (50,intersection_point[1]) #based on the cleft edges (only works, when the cleft opens to the left)
    front_mask = keep_contour_with_point(front_mask, check_point)

    # Smooth Mask Contour
    # Apply morphological operations to remove small noise
    kernel = np.ones((segpar["front_masksmoothing_ksize"], segpar["front_masksmoothing_ksize"]), np.uint8)
    front_mask = cv2.morphologyEx(front_mask, cv2.MORPH_OPEN, kernel)

    ### Erode the mask (includes inversion to erode in the right direction)
    front_mask = invert_mask(front_mask)
    kernel_size = segpar["front_erosion_ksize"]
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    front_mask = cv2.erode(front_mask, kernel, iterations=segpar["front_erosion_iters"])
    front_mask = invert_mask(front_mask)
    front_mask = keep_largest_edge_contour(front_mask) #this catches errors due to multiple masked regions after mask-smoothing!

    ### get the front contour
    front_contour = extract_contour_line(front_mask)

    return cleft_mask, cleft_contour, front_mask, front_contour, rep_lines, intersection_point

'''
############################################################################################
### Old Version of Segmentation ###
def SegementGrowthFront(image):
    # Apply Sobel filter to detect edges
    kernel_size_sobel = 3
    filtered_image = SobelFilter(image, kernel_size_sobel)

    # Smooth image
    kernel_size=(9,9)
    sigma=3
    filtered_image = GaussFilter(filtered_image, kernel_size, sigma)

    check_point = (100,500)

    threshold = 20
    binary_mask = IntensitySegmentation(filtered_image, threshold)
    binary_mask = invert_mask(binary_mask)
    binary_mask = keep_contour_with_point(binary_mask, check_point)
    contour_line = extract_contour_line(binary_mask)

    return binary_mask, filtered_image, contour_line


def SegementCleft(image):
    # Smooth image
    kernel_size=(15,15)
    sigma=5
    filtered_image = GaussFilter(image, kernel_size, sigma)

    threshold = 140
    binary_mask = IntensitySegmentation(filtered_image, threshold)
    binary_mask = keep_largest_edge_contour(binary_mask)
    contour_line = extract_contour_line(binary_mask)
    binary_mask = invert_mask(binary_mask)


    return binary_mask, filtered_image, contour_line
#############################################################################################
'''