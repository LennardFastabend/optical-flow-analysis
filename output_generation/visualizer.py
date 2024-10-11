import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
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
        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale = 1, width=0.001, color=colormap(norm(colors))) #scale = 0.2
        fig.savefig(self.output_path / filename, dpi=600)   # save the figure to file
        plt.close(fig)    # close the figure window

    def saveDeformationMap(self, DeformationMap, min, max, title, filename):
        fig, ax = plt.subplots(1)
        plt.imshow(DeformationMap, cmap='plasma', vmin=min, vmax=max)
        plt.title(title)
        #colorbar = plt.colorbar()
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
        plt.plot(cleft_contour_line[:, 0], cleft_contour_line[:, 1], marker='.', markersize=1, color='green', linestyle='-', linewidth=1)
        plt.savefig(self.output_path / filename, dpi=600)   # save the figure to file
        plt.close()    # close the figure window

    def saveGeometricQuantificationBinnedStatistics(self,df, bin_size, max_shown_distance, min_shown_displacement, max_shown_displacement, title, filename):
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
        plt.ylim(min_shown_displacement,max_shown_displacement)

        plt.xlabel('Distance Bin')
        plt.ylabel('Mean Displacement')
        plt.title(title)
        plt.xticks(rotation=270)  # Rotate x-axis labels for better readability
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_path / filename, dpi=600)   # save the figure to file
        plt.close()    # close the figure window



    def saveGeometricQuantificationCumulativeBinnedStatistics(self, df_list, bin_size, max_shown_distance, min_shown_displacement, max_shown_displacement, title, filename):
        # Combine all the dataframes into one
        combined_df = pd.concat(df_list, ignore_index=True)

        # Define the bins for the distance values
        min_distance = combined_df['distance'].min()
        max_distance = combined_df['distance'].max()
        bins = np.arange(min_distance, max_shown_distance + bin_size, bin_size)

        # Bin the distance values
        combined_df['binned_distance'] = pd.cut(combined_df['distance'], bins, include_lowest=True)

        # Calculate the mean and standard deviation for each binned distance
        binned_stats = combined_df.groupby('binned_distance')['displacement'].agg(['mean', 'std']).reset_index()

        # Prepare bin centers for plotting, instead of bin labels
        bin_centers = bins[:-1] + bin_size / 2

        # Plot the binned data with error bars
        plt.figure(figsize=(10, 5))
        plt.errorbar(
            bin_centers,                 # X-axis: Bin centers
            binned_stats['mean'],         # Y-axis: Mean displacement
            yerr=binned_stats['std'],     # Error bars: Standard deviation
            fmt='o',                      # Markers for the data points
            ecolor='r',                   # Error bar color
            capsize=5,                    # Cap size for error bars
            label='Mean Displacement with Std. Dev.'
        )

        # Set consistent axis limits
        plt.xlim(0, max_shown_distance)
        plt.ylim(min_shown_displacement, max_shown_displacement)

        plt.xlabel('Distance to Growth Front')
        plt.ylabel('Mean Displacement')
        plt.title(title)
        plt.xticks(rotation=270)  # Rotate x-axis labels for better readability
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_path / filename, dpi=600)   # Save the figure to file
        plt.close()    # Close the figure window

    def saveGeometricQuantificationCumulativePercentileBands(self, df_list, smoothing_winsize, max_shown_distance, min_shown_displacement, max_shown_displacement, title, filename):
        # Combine all the dataframes into one
        combined_df = pd.concat(df_list, ignore_index=True)

        # Sort the dataframe by distance
        combined_df = combined_df.sort_values(by='distance')

        # Calculate percentiles
        percentiles = combined_df.groupby('distance')['displacement'].quantile([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).unstack()

        # Prepare distance values (x-axis)
        distances = percentiles.index

        # Apply a rolling mean for smoothing the percentiles
        smoothed_percentiles = percentiles.rolling(window=smoothing_winsize, center=True, min_periods=1).mean()

        # Plot the data
        plt.figure(figsize=(10, 5))

        # Define the transparency (alpha) for the percentiles
        alpha_val = 0.5

        # Plot the outer percentiles (1st to 99th)
        plt.fill_between(distances, smoothed_percentiles[0.01], smoothed_percentiles[0.99], color='yellow', alpha=alpha_val, label='1st-99th Percentile')

        # Plot the outer percentiles (10th to 90th)
        plt.fill_between(distances, smoothed_percentiles[0.1], smoothed_percentiles[0.9], color='orange', alpha=alpha_val, label='10th-90th Percentile')

        # Plot the 25th to 75th percentiles
        plt.fill_between(distances, smoothed_percentiles[0.25], smoothed_percentiles[0.75], color='red', alpha=alpha_val, label='25th-75th Percentile')

        # Plot the 50th percentile (median)
        plt.plot(distances, smoothed_percentiles[0.5], color='black', label='50th Percentile (Median)', linewidth=2)

        # Set axis limits
        plt.xlim(0, max_shown_distance)
        plt.ylim(min_shown_displacement, max_shown_displacement)

        # Labels and title
        plt.xlabel('Distance to Growth Front')
        plt.ylabel('Displacement')
        plt.title(title)

        # Grid and legend
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save the plot to file
        plt.savefig(self.output_path / filename, dpi=600)
        plt.close()  # Close the figure window




    def saveGeometricQuantificationScatterPlot(self, df, max_shown_distance, min_shown_displacement, max_shown_displacement, title, filename, c='#1f77b4'):
        # Create the scatter plot
        plt.figure(figsize=(10, 5))
        plt.scatter(df['distance'], df['displacement'], s=1.5, marker='.', alpha=0.25, color=c) #label='Displacement of Pixel in Dependency of the Distance to the Growth Front ')

        # Set consistent axis limits
        plt.xlim(0, max_shown_distance)
        plt.ylim(min_shown_displacement, max_shown_displacement)

        plt.xlabel('Distance to Growth Front')
        plt.ylabel('Displacement')
        plt.title(title)
        plt.grid(True)
        #plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_path / filename, dpi=600)  # save the figure to file
        plt.close()  # close the figure window

    def saveCumulativeGeometricQuantificationScatterPlot(self, df_list, max_shown_distance, min_shown_displacement, max_shown_displacement, title, filename, c='blue'):
        plt.figure(figsize=(10, 5))
        
        # Loop through each dataframe in the list and plot the data
        for df in df_list:
            plt.scatter(df['distance'], df['displacement'], s=1.5, marker='.', alpha=0.005, color=c)

        # Set consistent axis limits
        plt.xlim(0, max_shown_distance)
        plt.ylim(min_shown_displacement, max_shown_displacement)

        plt.xlabel('Distance to Growth Front')
        plt.ylabel('Displacement')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_path / filename, dpi=600)  # save the figure to file
        plt.close()  # close the figure window

    def saveCumulativeGeometricQuantificationHeatMap(self, df_list, max_shown_distance, min_shown_displacement, max_shown_displacement, title, filename, bins=1000, cmap='jet'):
        plt.figure(figsize=(10, 5))
        
        # Initialize arrays to hold all distance and displacement data
        all_distances = np.array([])
        all_displacements = np.array([])
        
        # Collect data from all dataframes
        for df in df_list:
            all_distances = np.append(all_distances, df['distance'])
            all_displacements = np.append(all_displacements, df['displacement'])
        
        # Create 2D histogram with the specified number of bins
        heatmap, xedges, yedges = np.histogram2d(all_distances, all_displacements, bins=bins, range=[[0, max_shown_distance], [min_shown_displacement, max_shown_displacement]])
        
        # Plot heatmap with logarithmic color scale
        plt.imshow(heatmap.T, extent=[0, max_shown_distance, min_shown_displacement, max_shown_displacement], origin='lower', aspect='auto', cmap=cmap, norm=LogNorm(vmin=1, vmax=100))
        
        plt.colorbar(label='Density (log scale)')
        plt.xlabel('Distance to Growth Front')
        plt.ylabel('Displacement')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_path / filename, dpi=600)  # save the figure to file
        plt.close()  # close the figure window


    def saveDirectionalGeometricQuantificationScatterPlot(self, df, max_shown_distance, max_shown_displacement, title, filename, displacement_type='x'):
        if displacement_type not in ['x', 'y']:
            raise ValueError("displacement_type must be either 'x' or 'y'")
        
        # Select the appropriate displacement column and set color
        displacement_column = f'{displacement_type}_displacement'
        color = 'red' if displacement_type == 'x' else 'green'
        
        # Create the scatter plot
        plt.figure(figsize=(10, 5))
        plt.scatter(df['distance'], df[displacement_column], s=1.5, marker='.', alpha=0.25, color=color)

        # Set consistent axis limits
        plt.xlim(0, max_shown_distance)
        plt.ylim(-max_shown_displacement, max_shown_displacement)

        plt.xlabel('Distance to Growth Front')
        plt.ylabel(f'{displacement_type.upper()} Displacement')
        plt.title(f'{displacement_type.upper()}-{title}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_path / f'{filename}_{displacement_type}.png', dpi=600)  # save the figure to file
        plt.close()  # close the figure window

    def saveCleftEdgeFlow(self, pos_l, pos_w, normal_def, parallel_def, filename, min_l, max_l, min_w, max_w, min_def=-10, max_def=10):
        # Create a mask for filtering based on the min and max values
        mask = (pos_l >= min_l) & (pos_l <= max_l) & (pos_w >= min_w) & (pos_w <= max_w)
        # Apply the mask to pos_l, pos_w, and deformation
        filtered_l = pos_l[mask]
        filtered_w = pos_w[mask]
        filtered_deformation_normal = normal_def[mask]
        filtered_deformation_parallel = parallel_def[mask]

        ### Plotting
        # Calculate the aspect ratio based on the range of x and y limits
        aspect_ratio = (max_l - min_l) / (max_w - min_w)

        # Assuming you want the width of the figure to be 10 inches
        width_inch = 10
        height_inch = width_inch / aspect_ratio  # Ensuring the l axis is longer than the w axis

        # Create a figure with two subplots stacked vertically
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width_inch, height_inch * 2))  # height_inch * 2 for stacking subplots

        # Plot in the first subplot
        sc1 = ax1.scatter(filtered_l, filtered_w, c=filtered_deformation_normal, cmap='seismic', s=10, vmin=min_def, vmax=max_def)
        ax1.set_xlabel('l', fontsize=14)
        ax1.set_ylabel('w', fontsize=14)
        ax1.set_title('Normal Deformation in l-w Coordinate System', fontsize=16)
        ax1.set_xlim(min_l, max_l)
        ax1.set_ylim(min_w, max_w)
        ax1.plot([min_l, max_l], [0, 0], color='black', linewidth=2)
        ax1.plot([min_l, min_l], [min_w, max_w], color='black', linewidth=2)
        ax1.grid(True)
        fig.colorbar(sc1, ax=ax1, label='Normal Deformation')

        # Plot in the second subplot
        sc2 = ax2.scatter(filtered_l, filtered_w, c=filtered_deformation_parallel, cmap='seismic', s=10, vmin=min_def, vmax=max_def)
        ax2.set_xlabel('l', fontsize=14)
        ax2.set_ylabel('w', fontsize=14)
        ax2.set_title('Parallel Deformation in l-w Coordinate System', fontsize=16)
        ax2.set_xlim(min_l, max_l)
        ax2.set_ylim(min_w, max_w)
        ax2.plot([min_l, max_l], [0, 0], color='black', linewidth=2)
        ax2.plot([min_l, min_l], [min_w, max_w], color='black', linewidth=2)
        ax2.grid(True)
        fig.colorbar(sc2, ax=ax2, label='Parallel Deformation')

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Save the plot
        plt.savefig(self.output_path / filename, dpi=600)  # save the figure to file
        plt.close()  # close the figure window

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