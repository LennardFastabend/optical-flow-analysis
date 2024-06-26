import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

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


def GaussFilter(image, kernel_size, sigma):
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

############################################################################################
    
def SegementGrowthFront(image):
    # Apply Sobel filter to detect edges
    kernel_size_sobel = 3
    filtered_image = SobelFilter(image, kernel_size_sobel)

    # Smooth image
    kernel_size=(9,9)
    sigma=3
    filtered_image = GaussFilter(filtered_image, kernel_size, sigma)

    check_point = (100,500)

    threshold = 10
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

    #check_point = (100,500)
    #if binary_mask[check_point[1], check_point[0]] == 0:
        #binary_mask = invert_mask(binary_mask)
        #binary_mask = remove_contour_with_point(binary_mask, check_point)
        #binary_mask = invert_mask(binary_mask)


    return binary_mask, filtered_image, contour_line