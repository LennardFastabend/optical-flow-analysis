import numpy as np
import openpiv.pyprocess
#import openpiv.validation
#import openpiv.filters
#import openpiv.scaling
#import cv2

import sys

def PIVStack(img_stack, par_dict, t0, tfin, dt=1):
    # Initialize an empty list to store the flow fields
    FlowField_list = []
    
    # Iterate over the specified time frames
    for t in range(t0, tfin):
        print('PIV at Time: ', t)
        img0 = img_stack[t, :, :]
        img1 = img_stack[t + dt, :, :]

        # Perform PIV using the extended_search_area_piv function
        u, v, sig2noise = openpiv.pyprocess.extended_search_area_piv(
            img0.astype(np.int32),
            img1.astype(np.int32),
            window_size=par_dict["window_size"],
            overlap=par_dict["overlap"],
            dt=dt,
            search_area_size=par_dict["window_size"],
            sig2noise_method = None #'peak2peak' or 'peak2mean', or None
        )

        # Create an output array with the same shape as the original images, filled with zeros
        flow_field = np.zeros((img0.shape[0], img0.shape[1], 2))  # Fill with zeros
        
        # Calculate the step size based on window size and overlap
        step_size = par_dict["window_size"] - par_dict["overlap"]
        
        # Place the calculated vectors in the appropriate positions
        for i in range(u.shape[0]):  # Iterate over the number of rows of vectors
            for j in range(u.shape[1]):  # Iterate over the number of columns of vectors
                y_pos = i * step_size
                x_pos = j * step_size
                
                # Assign the vector to the center of the window
                flow_field[y_pos, x_pos, 0] = u[i, j]  # X component
                flow_field[y_pos, x_pos, 1] = v[i, j]  # Y component

        # Append the flow field for this frame to the list
        FlowField_list.append(flow_field)

    # Stack all flow fields over time
    FlowField_stack = np.stack(FlowField_list, axis=0)  # Shape (t, height, width, 2)

    return FlowField_stack