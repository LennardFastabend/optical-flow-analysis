import numpy as np
import pandas as pd

def GeometricQuantification(defmap, tissue_mask, front_contour_line, front_center, dx=100):
    data = [] #list to store (distance, deformation) value pairs
    dx = 100 #maximum distance from growth front center for deformations to be considered
    # Find the coordinates of non-zero points in the mask
    masked_pixels = np.column_stack(np.where(tissue_mask > 0))
    # Iterate over masked tissue deformation
    for pixel in masked_pixels:
        y = pixel[0]
        x = pixel[1]
        if x >= front_center[0]-dx:
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