import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

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

def GeometricQuantificationDistanceMap(defmap, tissue_mask, distance_map, xmax_front, dx=100):
    '''
    Simple Geometric Quantification, that stores the deformation magnitude together with the distance to the growth front using a distance map
    '''
    data = []  # list to store (distance, deformation) value pairs
    dx = 100  # maximum distance from growth front center for deformations to be considered
    # Find the coordinates of non-zero points in the mask
    masked_pixels = np.column_stack(np.where(tissue_mask > 0))
    # Iterate over masked tissue deformation
    for pixel in masked_pixels:
        y = pixel[0]
        x = pixel[1]
        if x >= xmax_front - dx:
            # Access the precomputed distance from the distance_map
            min_distance = distance_map[y, x]
            # Access the pixel value in the image
            pixel_value = defmap[y, x]
            data.append((min_distance, pixel_value))

    # Convert the list of tuples to a pandas DataFrame
    df = pd.DataFrame(data, columns=['distance', 'displacement'])
    return df

def ComputeNormalVectorField(tissue_mask, front_mask):
    '''
    Compute the normal vector field and distance map for a given tissue mask and front mask.
    
    Parameters:
    tissue_mask (2D array): Binary mask of the tissue.
    front_mask (2D array): Binary mask of the growth front.

    Returns:
    normal_vectors (3D array): Normal vector field with shape (height, width, 2).
    distance_map (2D array): Distance map quantifying the shortest distance to the front_mask.
    '''
    # Initialize a 3D array to store normal vector components
    normal_vectors = np.zeros((tissue_mask.shape[0], tissue_mask.shape[1], 2), dtype=np.float32)

    # Compute the distance map and the indices of the nearest front points
    # The distance transform will be computed from the front mask
    # We invert the front mask to compute distances outside the front mask
    distance_map, indices = distance_transform_edt(~front_mask, return_indices=True)
    
    # Mask the distance map with the tissue mask to zero out distances outside the tissue
    distance_map = np.where(tissue_mask, distance_map, 0)

    # Find the coordinates of non-zero points in the mask
    masked_pixels = np.column_stack(np.where(tissue_mask > 0))
    
    # Iterate over masked tissue
    for pixel in masked_pixels:
        y = pixel[0]
        x = pixel[1]

        # Find the nearest front point
        nearest_front_point = indices[:, y, x]
        front_y, front_x = nearest_front_point

        # Calculate the normal vector pointing to the nearest front point
        normal_vector = np.array([front_x - x, front_y - y])
        #normal_vector = normal_vector / np.linalg.norm(normal_vector)

        # Store the normal vector components in the 3D array
        normal_vectors[y, x, 0] = normal_vector[0]
        normal_vectors[y, x, 1] = normal_vector[1]
    
    return normal_vectors, distance_map

def ComputeNormalAndParallelDisplacement(FlowField, normal_vectors):
    # Normalise Normalvectors
    norm = np.linalg.norm(normal_vectors, axis=2, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    normalized_normals = normal_vectors / norm
    
    # Compute component parallel to normalvector
    dot_products = np.sum(FlowField * normalized_normals, axis=2, keepdims=True)
    parallel_component = dot_products.squeeze()
    
    # compute parallel vector
    parallel_vectors = parallel_component[..., np.newaxis] * normalized_normals
    
    # Orthogonal vector
    orthogonal_vectors = FlowField - parallel_vectors
    
    # Berechnung des Vorzeichens der orthogonalen Komponente
    # Normalenvektor um 90° rotieren: [n_y, -n_x]
    rotated_normals = np.stack([-normalized_normals[..., 1], normalized_normals[..., 0]], axis=-1)
    orthogonal_component = np.sum(orthogonal_vectors * rotated_normals, axis=2)
    
    return orthogonal_component, parallel_component  #Note: parallel comp. is normal to growthfront, orth. comp. is paralell to growth front













'''
def DirectionalGeometricQuantification(FlowField, tissue_mask, front_contour_line, xmax_front, dx=100):
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

def ExtensiveGeometricQuantification(FlowField, tissue_mask, front_contour_line, xmax_front, dx=100):
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
'''