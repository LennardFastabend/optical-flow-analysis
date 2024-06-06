from data_reader.reader import reader
from output_generation.visualizer import visualizer
import data_analysis.optical_flow as opflow

from pathlib import Path
import numpy as np

root_dir = Path(r'C:\Users\lenna\Documents\GitHub\optical-flow-analysis') #path to repository
#input_dir = Path(r'data\PhaseContrastCleft\P01\raw_data\P08#39_live_191127_1812_MB-0003_W03-P01-1.zvi') #relative path to input data
input_dir = Path(r'data\PhaseContrastCleft\P01\input\P08#39_live_W03-P01.avi')
output_dir = Path(r'data\PhaseContrastCleft\P01\output')

input_reader = reader(root_dir, input_dir)
image_stack = input_reader.read_avi()
flowfield_stack = opflow.FlowFieldStack(image_stack, t0=0, tfin=50, dt=1)

#image_generator = visualizer(root_dir, output_dir/Path('images'))
flowfield_generator = visualizer(root_dir, output_dir/Path('flowfields'))
defmap_generator = visualizer(root_dir, output_dir/Path('defmap'))

#'''
dt=3
Tmax = 50
for T in np.arange(0,Tmax-dt):
    meanflowfield = opflow.MeanFlowField(flowfield_stack[T:T+dt,...])
    defmap = opflow.calculateMagnitude(meanflowfield)
    #image_generator.saveImage(image_stack[T,...], title='Cleft at Time: '+str(T), filename='Image'+str(T))
    flowfield_generator.saveFlowField(image_stack[T,...], meanflowfield, title='FlowField at Time: '+str(T)+'-'+str(T+dt), filename='FlowField'+str(T), step=20, epsilon=0)
    defmap_generator.saveDeformationMap(defmap, min=0, max=10, title='DefMap at Time: '+str(T)+'-'+str(T+dt), filename='DefMap'+str(T))
#'''

#image_generator.create_video(fps=10)
flowfield_generator.create_video(fps=10)
defmap_generator.create_video(fps=10)