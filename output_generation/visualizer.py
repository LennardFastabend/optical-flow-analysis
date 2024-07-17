import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
import cv2
import os
from pathlib import Path

import numpy as np
import pandas as pd

class visualizer:
    def __init__(self, root_dir, output_dir):
        self.output_path = root_dir / output_dir
        print('Initialized Visualizer with Path:', self.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def saveImage(self, img, title, filename):
        fig, ax = plt.subplots(1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255) #origin="lower"
        plt.title(title)
        #plt.colorbar()
        #plt.xlabel('x')
        #plt.ylabel('y')
        plt.axis('off')
        fig.savefig(self.output_path / filename, dpi=600)   # save the figure to file
        plt.close(fig)    # close the figure window

    def saveFlowField(self, prev_img, FlowField, title, filename, step=20, epsilon=0):
        Fx = FlowField[...,0]
        Fy = FlowField[...,1]

        ydim = prev_img.shape[0]
        xdim = prev_img.shape[1]

        idx_x = np.arange(xdim)
        idx_y = np.arange(ydim)
        idx_x,idx_y = np.meshgrid(idx_x, idx_y)

        #only plot every step-th vector (set every other entry to zero)
        for i in range(xdim):
            for j in range(ydim):
                if (i%step==0 and j%step==0):
                    pass
                else:   
                    Fx[j][i] = 0
                    Fy[j][i] = 0

        #only plot vectors that have a length < epsilon
        mask = np.logical_or(abs(Fx) > epsilon, abs(Fy) > epsilon)

        #compute positions and lengths of all non-zero vectors
        X = idx_x[mask]
        Y = idx_y[mask]
        U = Fx[mask]
        V = Fy[mask]

        #color of quiver
        colors = np.arctan2(V, U)
        norm = Normalize()
        norm.autoscale(colors)
        colormap = cm.hsv

        fig, ax = plt.subplots(1)
        plt.imshow(prev_img, cmap='gray', vmin=0, vmax=255) #origin="lower"
        plt.title(title)
        #plt.colorbar()
        #plt.xlabel('x')
        #plt.ylabel('y')
        plt.axis('off')
        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale = 0.2, width=0.001, color=colormap(norm(colors))) #, color='r'
        fig.savefig(self.output_path / filename, dpi=600)   # save the figure to file
        plt.close(fig)    # close the figure window

    def saveDeformationMap(self, DeformationMap, min, max, title, filename):
        fig, ax = plt.subplots(1)
        plt.imshow(DeformationMap, cmap='plasma', vmin=min, vmax=max)
        plt.title(title)
        colorbar = plt.colorbar()
        #colorbar.set_label('µm/h', labelpad=-35, y=1.08, rotation=0)
        #colorbar.set_ticks([0, 1, 2, 3, 4, 5])
        #colorbar.set_ticklabels([0, 0.65, 1.30, 1.95, 2.60, 3.25])
        #plt.xlabel('x')
        #plt.ylabel('y')
        plt.axis('off')
        fig.savefig(self.output_path / filename, dpi=600)   # save the figure to file
        plt.close(fig)    # close the figure window


    '''
    Visualise a Flow Field-stack by calculate mean displacements ofer the given FlowField_stack and
    converting the (mean) vector for each pixel into a color based on the direction/angle and the magnitude of the vector 
    (convert vector to HSV and hsv to rgb)
    angle of vector -> hue (Farbton)
    magnitude of vector -> saturation and brightness
    max_magnitude defines the maximum 
    '''
    def saveHSVcolormap(self, max_magnitude, filename):
        def VectorToRGB(angle, magnitude, max_magnitude):
            angle = (angle + np.pi) / (2 * np.pi)  # Normalize angle to [0, 1]
            angle = 1 - angle  # Flip the colorwheel on the x-axis

            magnitude = np.clip(magnitude / max_magnitude, 0, 1)  # Normalize and clip magnitude to [0, 1]

            rgb = matplotlib.colors.hsv_to_rgb((angle, magnitude, magnitude))

            return rgb
        
        #plot: hsv colormap
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))

        n = 200
        t = np.linspace(-np.pi, np.pi, n)
        r = np.linspace(0, max_magnitude, n)
        tg, rg = np.meshgrid(t, r)

        # Create an RGB colormap based on vector angles and magnitudes
        c = np.array([VectorToRGB(t, r, max_magnitude) for t, r in zip(tg.flatten(), rg.flatten())])
        cv = c.reshape((n, n, 3))

        m = ax.pcolormesh(t, r, cv[:, :, 1], color=c, shading='auto')
        m.set_array(None)

        ax.tick_params(axis='y', colors='#C6C6C6')

        # Save the figure to a file
        fig.savefig(self.output_path / filename, dpi=600)
        plt.close(fig)

    def saveDeformationMapRGB(self, FlowField, max_magnitude, title, filename):
        # Define the FlowField-components in x- and y-direction
        Fx = FlowField[:, :, 0]
        Fy = FlowField[:, :, 1]

        # Calculate the magnitudes of the vectors
        magnitudes = np.sqrt(np.square(Fx) + np.square(Fy))
        magnitudes = np.clip(magnitudes / max_magnitude, 0, 1)  # Normalize magnitude and clip to [0, 1]
        
        # Calculate the angles of all vectors
        angles = np.arctan2(Fy, Fx)
        angles = (angles + np.pi) / (2 * np.pi)  # Normalize angles to [0, 1]

        # Define a corresponding image in HSV colorspace
        hsv_image = np.zeros((FlowField.shape[0], FlowField.shape[1], 3))
        hsv_image[..., 0] = angles  # Hue
        hsv_image[..., 1] = magnitudes  # Saturation
        hsv_image[..., 2] = magnitudes  # Value

        # Convert the HSV image to RGB
        rgb_image = matplotlib.colors.hsv_to_rgb(hsv_image)

        fig, ax = plt.subplots(1)
        plt.imshow(rgb_image)
        plt.title(title)
        plt.axis('off')

        # Save the figure to a file
        fig.savefig(self.output_path / filename, dpi=600)
        plt.close(fig)
    
    def saveDivergence(self, div, title, filename):
        # Determine the maximum absolute value in div
        max_abs_value = np.max(np.abs(div))
        # Set vmin and vmax to be symmetric around zero
        vmin = -max_abs_value/4
        vmax = max_abs_value/4

        fig, ax = plt.subplots(1)
        plt.imshow(div, cmap='seismic', vmin=vmin, vmax=vmax)
        plt.title(title)
        
        cbar = plt.colorbar()
        '''
        cbar.set_label('h$^{-1}$', labelpad=-40, y=1.06, rotation=0)
        '''

        # Turn off axis ticks and labels
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        # Show only the frame around the data
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)

        fig.savefig(self.output_path / filename, dpi=600)   # save the figure to file
        plt.close(fig)    # close the figure window

    def saveSegmentationMasks(self, image, front_contour_line, cleft_contour_line, title, filename):
        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.imshow(image, cmap='gray')
        plt.plot(front_contour_line[:, 0], front_contour_line[:, 1], marker='.', markersize=1, color='red', linestyle='-', linewidth=1)
        plt.plot(cleft_contour_line[:, 0], cleft_contour_line[:, 1], marker='.', markersize=1, color='orange', linestyle='-', linewidth=1)
        plt.savefig(self.output_path / filename, dpi=600)   # save the figure to file
        plt.close()    # close the figure window

    def saveGeometricQuantification(self,df, bin_size, max_shown_distance, max_shown_displacement, title, filename):
        # Define the bins for the distance values
        min_distance = df['distance'].min()
        max_distance = df['distance'].max()
        bins = np.arange(min_distance, max_distance + bin_size, bin_size)

        # Bin the distance values
        df['binned_distance'] = pd.cut(df['distance'], bins, include_lowest=True)

        # Calculate the mean and standard deviation for each binned distance
        binned_stats = df.groupby('binned_distance')['displacement'].agg(['mean', 'std']).reset_index()

        # Plot the binned data with error bars
        plt.figure(figsize=(10, 5))
        plt.errorbar(
            binned_stats['binned_distance'].astype(str),
            binned_stats['mean'],
            yerr=binned_stats['std'],
            fmt='o',
            ecolor='r',
            capsize=5,
            label='Mean Displacement with Std. Dev.'
        )

        # Set consistent axis limits
        plt.xlim(-1,max_shown_distance/bin_size)
        plt.ylim(0,max_shown_displacement)

        plt.xlabel('Distance Bin')
        plt.ylabel('Mean Displacement')
        plt.title(title)
        plt.xticks(rotation=270)  # Rotate x-axis labels for better readability
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_path / filename, dpi=600)   # save the figure to file
        plt.close()    # close the figure window

    def create_video(self, fps=30, interval=1):
        def natural_sort_key(s):
            import re
            return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

        # Get the list of image files in the input folder
        image_files = sorted([f for f in os.listdir(self.output_path) if f.endswith('.png')], key=natural_sort_key)

        # Get the first image to get the width and height information
        first_image = cv2.imread(os.path.join(self.output_path, image_files[0]))
        height, width, _ = first_image.shape

        # Create a video writer object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
        video_writer = cv2.VideoWriter(str(self.output_path/Path('Animation'+str(fps)+'fps.mp4')), fourcc, fps, (width, height))

        # Write selected images to the video
        for i, image_file in enumerate(image_files):
            if i % interval == 0:
                image_path = os.path.join(self.output_path, image_file)
                frame = cv2.imread(image_path)
                video_writer.write(frame)

        # Release the video writer object
        video_writer.release()