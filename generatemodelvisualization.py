import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import matplotlib.pyplot as plt
from torchview import draw_graph
import os
import keras
import onnx2keras
import onnx
from onnx2keras import onnx_to_keras
from keras.api.models import Model
from keras.api.layers import Input, Conv2D, ReLU, MaxPooling2D
from keras.api.layers import Flatten, Dropout, Dense, Softmax
from keras.api.layers import TorchModuleWrapper


os.environ["PYTORCH_JIT"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 4096)

        self.conv1 = nn.Conv2d(3, 3, 5)
        self.conv2 = nn.Conv2d(3, 1, 3)

        self.fc3 = nn.Linear(784, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_dim)

        
    def forward(self, x):
        out = F.relu(self.fc1(x))

        out = F.relu(self.fc2(out))

        out = out.view(-1, 64, 64)

        out = self.conv1(out)

        # print(out.shape)
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)

        out = out.view(-1, 28, 28).flatten()
        out = self.fc3(out)
        out = self.fc4(out).view(-1, 41, 41)

        return out

class NeuralNetKeras(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = TorchModuleWrapper(nn.Linear(41, 1681))
        self.fc2 = TorchModuleWrapper(nn.Linear(1681, 4096))

        self.conv1 = TorchModuleWrapper(nn.Conv2d(3, 3, 5))
        self.conv2 = TorchModuleWrapper(nn.Conv2d(3, 1, 3))

        self.fc3 = TorchModuleWrapper(nn.Linear(784, 1681))
        self.fc4 = TorchModuleWrapper(nn.Linear(1681, 1681))

        
    def call(self, x):
        out = F.relu(self.fc1(x))

        out = F.relu(self.fc2(out))

        out = out.view(-1, 64, 64)

        out = self.conv1(out)

        # print(out.shape)
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)

        out = out.view(-1, 28, 28).flatten()
        out = self.fc3(out)
        out = self.fc4(out).view(-1, 41, 41)

        return out
# model = NeuralNet(41, 1681, 1681).to(device)

# model.load_state_dict(torch.load('model.pth'))
# model.to(device)

# model.eval()

# model = NeuralNetKeras()
# model.build((3, 41, 41))
# model.compile(loss='mean_absolute_error', 
#          optimizer='Adam',
#          metrics=['accuracy'])

# print(model.summary())
# Load ONNX model
onnx_model = onnx.load('topologyOptimizationModel.onnx')
dot_img_file = './dataforpaper/modelgraph/modelgraph.png'
# keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

# Call the converter (input - is the main model input name, can be different for your model)
k_model = onnx_to_keras(onnx_model, ['input'])
k_model.build((3, 41, 41))
k_model.compile(loss='mean_absolute_error', 
         optimizer='Adam',
         metrics=['accuracy'])
print(k_model.summary())
# INPUTS_LIST = []

# def getPermutedListOfSix():
#     initialPermutation = itertools.product(range(2), repeat=2)
#     resultList = [i for i in [*initialPermutation] if sum(i) > 0]
#     # print(resultList)
#     return resultList

# def parabolicFlowProfileBC(boundary_value, BC_Array):
#     dim = 41.
#     l = dim/6.0
#     gbar = 1.0
#     coordinate_value = boundary_value + 1
#     if BC_Array[0] == 1 and (dim/4 - l/2) < coordinate_value < (dim/4 + l/2):
#         t = coordinate_value - dim/4
#         return gbar * (1 - (2*t/l)**2)
#     if BC_Array[1] == 1 and ((3 * dim)/4 - l/2) < coordinate_value < ((3 * dim)/4 + l/2):
#         t = coordinate_value - (3 * dim)/4
#         return gbar * (1 - (2*t/l)**2) 
#     return 0

# def generateBCArrayNumpy(BCFunction, inputArray, outputArray):
#     in_BC_Array = np.zeros(41) # using 40x40 numpy arrays
#     out_BC_Array = np.zeros(41)
#     # print(in_BC_Array)
#     for i in range(41):
#         in_BC_Array[i] = BCFunction(i, inputArray)
#         out_BC_Array[i] = BCFunction(i, outputArray)
#     return np.array([in_BC_Array, out_BC_Array])

# def generateRandomInputs():
#     result = []
#     values = getPermutedListOfSix()
#     values2 = getPermutedListOfSix()
#     for i in range(len(values)):
#         for j in range(len(values2)):
#             result.append((generateBCArrayNumpy(parabolicFlowProfileBC, values[i], values2[j]), values[i], values2[i]))
#         # break
#         print(values[i])
#     return result

# inputs = generateRandomInputs()



# # volfrac = 22
# for i in inputs:
#     for volfrac in range(20, 21):
#         boundaryCondition = i[0]
#         in_barrier = i[1]
#         out_barrier = i[2]
#         InputResult = torch.tensor(np.array([[boundaryCondition[0]], [boundaryCondition[1]], np.full((1, 41), volfrac)])).to(device).to(torch.float32)
#         # modelInput = torch.cat((topologyLabels, volFrac), -1).to(device)
#         output = model(InputResult)
#         output = output.cpu().detach().numpy()
#         # torchviz.make_dot(model(InputResult), params=dict(model.named_parameters())).render("./modelvisualization", format="png")
#         # writer.add_graph(model, InputResult)
#         # writer.close()
#         model_graph = draw_graph(model, InputResult)
#         model_graph.visual_graph.render("./modelvisualization", format="png")
#         # hl_graph.save("./modelvisualization/", format="png")
#         # torch.onnx.export(model, InputResult, 'topologyOptimizationModel.onnx', input_names=["in_features"], output_names=["logits"])
#         # output = np.reshape(output, (41, 41))
#         # plt.imshow(output)
#         # print(i[1])
#         # strVal = '_'.join([str(j) for j in i[1]]) + "-" + '_'.join([str(j) for j in i[2]])
#         # plt.imsave(f'./modeloutputs/modeltest-{volfrac}-{strVal}.png', output)

