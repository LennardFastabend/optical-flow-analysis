from data_reader.reader import reader
from output_generation.visualizer import visualizer
import data_analysis.optical_flow as opflow
#import data_analysis.geometric_quantification as geoquant
#from data_segmentation.segmentation import Segmentation

import sys

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import time

start_time = time.time()



root_dir = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis') #path to repository
input_dir = Path(r'data\LaserCutting\input\S1B1\registered_video.avi') #"C:\Users\lenna\Documents\GitHub\optical-flow-analysis\data\LaserCutting\input\S1B1\registered_video.avi"
output_dir = Path(r'data\LaserCutting\S1B1\Aligned')
input_reader = reader(root_dir, input_dir)
image_stack = input_reader.read_avi()
t, y, x = image_stack.shape

print(image_stack.shape)

image_stack = image_stack[:, 50:y-50, 30:x-600] #crop the image


'''
### Save Original Video
out = cv2.VideoWriter('original_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (x, y))

for i in range(t):
    # Convert image to uint8 format
    frame = cv2.normalize(image_stack[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

out.release()
'''



'''
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Initial Image')
plt.imshow(image_stack[240,...], cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Final Image')
plt.imshow(image_stack[640,...], cmap='gray')

plt.show()
'''

### Alignment
'''
reference_image = image_stack[500, :, :]
image_stack = image_stack[40:840:1, ...] #image_stack[270:670:5, ...] # only consider frames that dont require alignment (so far only every 10th frame)

t, y, x = image_stack.shape
print(image_stack.shape)

### Image Registration:
from skimage import io
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

# Initialize an empty array to hold the registered images
registered_stack = np.zeros_like(image_stack)

# Iterate over each frame in the time series
for i in range(t):
    current_image = image_stack[i, :, :]
    
    # Compute the translation needed to align current image with the reference
    shift_estimation, error, _ = phase_cross_correlation(reference_image, current_image)
    
    # Apply the estimated shift to the current image
    registered_image = shift(current_image, shift_estimation, mode='constant', cval=0)
    
    # Store the registered image in the stack
    registered_stack[i, :, :] = registered_image



# Example: Save as a video using OpenCV
out = cv2.VideoWriter('registered_video.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (x, y))

for i in range(t):
    # Convert image to uint8 format
    frame = cv2.normalize(registered_stack[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

out.release()
'''






image_stack = image_stack[0:800:10, ...]


dT=1
Tmax = 80

farneback_parameters = {"pyr_scale": 0.5,
                        "levels": 3,
                        "winsize": 5,#15,
                        "iterations": 3,
                        "poly_n": 5,
                        "poly_sigma": 1.2,
                        "flags": 0}

print('Start Farneback Analysis')
flowfield_stack = opflow.FlowFieldStack(image_stack, farneback_parameters, t0=0, tfin=Tmax-1, dt=1)
print('Farneback Analysis Finished')

image_generator = visualizer(root_dir, output_dir/Path('images'))
#flowfield_generator = visualizer(root_dir, output_dir/Path('flowfields'))
defmap_generator = visualizer(root_dir, output_dir/Path('defmap'))
defmapRGB_generator = visualizer(root_dir, output_dir/Path('defmapRGB'))
#save the color map for the rgb_defmap seperately (after video generation)
max_magnitude = 0.5


for T in np.arange(0,Tmax-dT,1):
    print(T)

    image = image_stack[T,...]

    # Calculate a Deformation Map
    meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
    defmap = opflow.calculateMagnitude(meanflowfield)
    image_generator.saveImage(image_stack[T,...], title='Cleft at Time: '+str(T), filename='Image'+str(T))
    defmap_generator.saveDeformationMap(defmap, min=0, max=max_magnitude, title='DefMap at Time: '+str(T)+'-'+str(T+dT), filename='DefMap'+str(T))
    defmapRGB_generator.saveDeformationMapRGB(meanflowfield, max_magnitude, title='DefMap at Time: '+str(T)+'-'+str(T+dT), filename='DefMap'+str(T))
    #flowfield_generator.saveFlowField(image_stack[T,...], meanflowfield, title='FlowField at Time: '+str(T)+'-'+str(T+dT), filename='FlowField'+str(T), step=20, epsilon=0)


print("--- Runtime: %s seconds ---" % (time.time() - start_time))
print()

print('Generate Videos')
image_generator.create_video(fps=10)
#flowfield_generator.create_video(fps=10)
defmap_generator.create_video(fps=10)
defmapRGB_generator.create_video(fps=10)

defmapRGB_generator.saveHSVcolormap(max_magnitude, filename='colormap')