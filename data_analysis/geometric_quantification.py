import numpy as np
import pandas as pd

def GeometricQuantification(defmap, tissue_mask, front_contour_line, xmax_front, dx=100):
    '''
    Simple Geometric Quantification, that stores the deformation magnitude together with the distance to the growth front
    '''
    data = [] #list to store (distance, deformation) value pairs
    dx = 100 #maximum distance from growth front center for deformations to be considered
    # Find the coordinates of non-zero points in the mask
    masked_pixels = np.column_stack(np.where(tissue_mask > 0))
    # Iterate over masked tissue deformation
    for pixel in masked_pixels:
        y = pixel[0]
        x = pixel[1]
        if x >= xmax_front-dx:
            # Calculate the Euclidean distance from the pixel to each point in the contour
            distances = np.sqrt(np.sum((front_contour_line - [x,y]) ** 2, axis=1))
            # Find the minimum distance
            min_distance = np.min(distances)
            # Access the pixel value in the image
            pixel_value = defmap[y, x]
            data.append((min_distance, pixel_value))

    # Convert the list of tuples to a pandas DataFrame
    df = pd.DataFrame(data, columns=['distance', 'displacement'])
    return df

def DirectionalGeometricQuantification(FlowField, tissue_mask, front_contour_line, xmax_front, dx=100):
    '''
    Extended Geometric Quantification, that stores the X and Y components of the deformation field together with the distance to the growth front and the coordinates of the given pixel
    '''
    # Define the FlowField-components in x- and y-direction
    Fx_component = FlowField[:, :, 0]
    Fy_component = FlowField[:, :, 1]
    data = [] #list to store (distance, deformation) value pairs
    dx = 100 #maximum distance from growth front center for deformations to be considered
    # Find the coordinates of non-zero points in the mask
    masked_pixels = np.column_stack(np.where(tissue_mask > 0))
    # Iterate over masked tissue deformation
    for pixel in masked_pixels:
        y = pixel[0]
        x = pixel[1]
        if x >= xmax_front-dx:
            # Calculate the Euclidean distance from the pixel to each point in the contour
            distances = np.sqrt(np.sum((front_contour_line - [x,y]) ** 2, axis=1))
            # Find the minimum distance
            min_distance = np.min(distances)
            # Access the pixel value in the image
            Fx = Fx_component[y, x]
            Fy = Fy_component[y, x]
            data.append((x,y,min_distance,Fx,Fy))

    # Convert the list of tuples to a pandas DataFrame
    df = pd.DataFrame(data, columns=['x','y','distance','x_displacement', 'y_displacement'])
    return df