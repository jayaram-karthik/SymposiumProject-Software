import numpy as np
import os
import matplotlib.pyplot as plt

imgdir = './topologyimages/'
topologydir = './neuralnetworkdata/TopologyOutput/'

for file in os.listdir(topologydir):
    path = topologydir + file
    topology = np.load(path)
    topology = np.array(topology, dtype=np.float32).reshape(41, 41)
    plt.imshow(topology, cmap='gray')
    plt.axis('off')
    plt.savefig(imgdir + file[:-4] + '.png', bbox_inches='tight', pad_inches=0)
