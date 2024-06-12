import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
import cv2
import os
from pathlib import Path

import numpy as np


class visualizer:
    def __init__(self, root_dir, output_dir):
        self.output_path = root_dir / output_dir
        self.output_path.mkdir(parents=True, exist_ok=True)

    def saveImage(self, img, title, filename):
        fig, ax = plt.subplots(1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255) #origin="lower"
        plt.title(title)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
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

    def saveDivergence(self, div, title, filename):
        fig, ax = plt.subplots(1)
        plt.imshow(div, cmap='seismic', vmin=-1, vmax=1)
        plt.title(title)
        ''''
        cbar = plt.colorbar()
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