import cv2
import numpy as np

'''
class opticalflow:
    def __init__(self, image_stack=None, flowfield_stack=None, divergence_stack=None):
        self.image_stack = image_stack
        self.flowfield_stack = flowfield_stack
        self.divergence_stack = divergence_stack
'''
'''
Calculate a FlowField Stack (one vector field for each pair of previous and next image)
shape = (t,y,x,component of vector)
'''
def FlowFieldStack(img_stack, t0, tfin, dt=1):
    FlowField_list = []
    for t in range(t0, tfin, dt):
        img0 = img_stack[t, :, :]
        img1 = img_stack[t+dt, :, :]
        flow = cv2.calcOpticalFlowFarneback(img0, img1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        FlowField_list.append(flow)

    FlowField_stack = np.stack(FlowField_list, axis=0)

    return FlowField_stack

'''
Retun a mean FlowField of a given FlowFieldStack
'''
def MeanFlowField(FlowField_stack):
    timesteps = FlowField_stack.shape[0]
    meanFlowField = np.sum(FlowField_stack, axis=0)/timesteps
    return meanFlowField

'''
Gaussian blurring of a given FlowField based on the defined Kernelsize
'''
def BlurFlowField(FlowField, Kernelsize):
    #return an unblurred Field if the Kernelsize is 0
    if Kernelsize == 0: 
        return FlowField
    
    else:
        Fx = FlowField[:,:,0]
        Fy = FlowField[:,:,1]

        #Blur the x- and y-components separatly
        BlurredFx = cv2.GaussianBlur(Fx, ksize=(Kernelsize,Kernelsize), sigmaX=0) #sigmaX = 0 -> sigma x and y are computed based on the given Kernelsize
        BlurredFy = cv2.GaussianBlur(Fy, ksize=(Kernelsize,Kernelsize), sigmaX=0)

        BlurredFlowField = np.stack((BlurredFx,BlurredFy), axis=-1)
        return BlurredFlowField

'''
Calculate the divergence of a given VectorField/FlowField
'''
def Divergence(FlowField):
    Fx = FlowField[:,:,0]
    Fy = FlowField[:,:,1]
    gradFx = np.gradient(Fx, axis=1) #calc. the secound order derivative of an array along the x-axis
    gradFy = np.gradient(Fy, axis=0) #calc. the secound order derivative of an array along the y-axis
    div = gradFx + gradFy
    return div

'''
Calculate the magnitude of displacement of a given FlowFieldStack
averaged by the given number of time steps
Function can be used on any length of FlowFieldStack in the time domain.
'''
def calculateMagnitude(FlowField):
    Fx = FlowField[:,:,0]
    Fy = FlowField[:,:,1]
    #calculate the magnitude of the vector field
    Magnitude = np.sqrt(np.square(Fx) + np.square(Fy))
    return Magnitude