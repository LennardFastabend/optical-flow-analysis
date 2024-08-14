import numpy as np

def normalized_line_and_normal_vectors(line_intersection, lower_border_intersection, upper_border_intersection):
    ### Definition of normalized Line and Normal Vectors for lower and upper line
    #define vectors that point from the image boundaries to th eline intersection
    upper_line_vector = line_intersection - upper_border_intersection
    lower_line_vector = line_intersection - lower_border_intersection
    # Normalize the vectors to length 1
    upper_line_vector = upper_line_vector / np.linalg.norm(upper_line_vector)
    lower_line_vector = lower_line_vector / np.linalg.norm(lower_line_vector)

    #define orthogonal vectors to the lines, pointing inward to the cleft
    upper_normal_vector = np.array([-upper_line_vector[1], upper_line_vector[0]])/8 #rotate +90°
    lower_normal_vector = np.array([lower_line_vector[1], -lower_line_vector[0]])/8 #rotate -90°
    # Normalize the vectors to length 1
    upper_normal_vector = upper_normal_vector / np.linalg.norm(upper_normal_vector)
    lower_normal_vector = lower_normal_vector / np.linalg.norm(lower_normal_vector)

    return upper_line_vector, upper_normal_vector, lower_line_vector, lower_normal_vector


def positional_vector_transformation(image, origin, line_vector, normal_vector):
        #Define a position vector p for each pixel, that points from the boundary intersection to the pixel (apply on the whole image)
        # Create a grid of pixel coordinates
        height, width = image.shape
        y, x = np.indices((height, width))  # y: row indices, x: column indices

        # Calculate vectors from the origin to each pixel
        vectors_x = x - origin[0]  # x-coordinates of vectors
        vectors_y = y - origin[1]  # y-coordinates of vectors

        # Combine the x and y components into a single array of shape (height, width, 2)
        positions = np.stack((vectors_x, vectors_y), axis=-1)

        #calculate the projections of the position vector onto the line and normal vector
        pos_w = np.sum(positions * normal_vector, axis=-1)
        pos_l = np.sum(positions * line_vector, axis=-1)

        return pos_l, pos_w

def deformation_projections(flowfield, line_vector, normal_vector):
    ### Calculate the local deformation projections that are parallel and normal to the cleft edge
    #calculate the scalar product of the local deformation vectors with the basis vectors:
    parallel_def = np.sum(flowfield * line_vector, axis=-1)
    normal_def = np.sum(flowfield * normal_vector, axis=-1)
    return parallel_def, normal_def