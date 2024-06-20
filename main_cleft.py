from data_reader.reader import reader
from output_generation.visualizer import visualizer
import data_analysis.optical_flow as opflow

import sys

from pathlib import Path
import numpy as np

root_dir = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis') #path to repository
#input_dir = Path(r'data\PhaseContrastCleft\P01\input\raw_data\P08#39_live_191127_1812_MB-0003_W03-P01-1-bottom.tif') #relative path to input data
input_dir = Path(r'data\PhaseContrastCleft\P01\input\P08#39_live_W03-P01.avi')
output_dir = Path(r'data\PhaseContrastCleft\P01\output_win5_dt3')

dT=3
Tmax = 100#image_stack.shape[0]-1

input_reader = reader(root_dir, input_dir)
image_stack = input_reader.read_avi()
#image_stack = input_reader.read_tif()

print(image_stack.shape)
print(type(image_stack))

sys.exit()
farneback_parameters = {"pyr_scale": 0.5,
                        "levels": 3,
                        "winsize": 5,#15,
                        "iterations": 3,
                        "poly_n": 5,
                        "poly_sigma": 1.2,
                        "flags": 0}

print('Start Farneback Analysis')
flowfield_stack = opflow.FlowFieldStack(image_stack, farneback_parameters, t0=0, tfin=Tmax, dt=1)
print('Farneback Analysis Finished')
print()

#image_generator = visualizer(root_dir, output_dir/Path('images'))
flowfield_generator = visualizer(root_dir, output_dir/Path('flowfields'))
defmap_generator = visualizer(root_dir, output_dir/Path('defmap'))
#div_generator = visualizer(root_dir, output_dir/Path('div'))

#'''
for T in np.arange(0,Tmax):
    print('Output Time:', T)
    meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dT,...])
    defmap = opflow.calculateMagnitude(meanflowfield)
    div = opflow.Divergence(meanflowfield)
    #image_generator.saveImage(image_stack[T,...], title='Cleft at Time: '+str(T), filename='Image'+str(T))
    flowfield_generator.saveFlowField(image_stack[T,...], meanflowfield, title='FlowField at Time: '+str(T)+'-'+str(T+dT), filename='FlowField'+str(T), step=20, epsilon=0)
    defmap_generator.saveDeformationMap(defmap, min=0, max=10, title='DefMap at Time: '+str(T)+'-'+str(T+dT), filename='DefMap'+str(T))
    #div_generator.saveDivergence(div, title='Divergence at Time: '+str(T)+'-'+str(T+dT), filename='Div'+str(T))
#'''
print()
print('Generate Videos')
#image_generator.create_video(fps=10)
flowfield_generator.create_video(fps=10)
defmap_generator.create_video(fps=10)
#div_generator.create_video(fps=10)