#from skimage import io
import czifile    
import cv2
import numpy as np
from skimage import io

class reader:
    def __init__(self, root_dir, input_dir):
        self.input_path = root_dir / input_dir

    '''
    Read in .CZI image stack and return it in the format: (t,y,x)
    '''
    def read_czi(self):
        #czi = czifile.CziFile(path) #czi.axes -> STCYX0
        img_stack = czifile.imread(self.input_path)
        img_stack = img_stack[0,:,0,:,:,0]
        return img_stack
    
    '''
    Read in .CZI image stack and return it in the format: (t,y,x)
    '''
    def read_zvi(self):
        with czifile.CziFile(self.input_path) as czi:
            # Get the image data as a numpy array
            img_stack = czi.asarray()

        return img_stack

    '''
    Read in .tif image stack and return it in the format: (t,y,x)
    '''
    def read_tif(self):
        img_stack = io.imread(self.input_path)
        return img_stack
    
    '''
    Read in .avi video and return it in the format: (t,y,x)
    '''
    def read_avi(self):
        # Open the video file
        cap = cv2.VideoCapture(str(self.input_path))
        # Check if video opened successfully
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.input_path}")
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Initialize the array to store frames
        video_array = np.empty((frame_count, frame_height, frame_width), np.uint8)
        # Read frames and store them in the array
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video_array[frame_idx] = gray_frame
            frame_idx += 1
        # Release the video capture object
        cap.release()
        return video_array