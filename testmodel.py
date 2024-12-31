import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import matplotlib.pyplot as plt
import time

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
        # print(out.shape)

        out = self.conv1(out)

        # print(out.shape)
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)

        out = out.view(-1, 28, 28)
        print(out.shape)
        
        out = out.flatten()
        out = self.fc3(out)
        out = self.fc4(out).view(-1, 41, 41)

        return out
    
model = NeuralNet(41, 1681, 1681).to(device)

model.load_state_dict(torch.load('model.pth'))
model.to(device)

model.eval()
INPUTS_LIST = []

def getPermutedListOfSix():
    initialPermutation = itertools.product(range(2), repeat=2)
    resultList = [i for i in [*initialPermutation] if sum(i) > 0]
    # print(resultList)
    return resultList

def parabolicFlowProfileBC(boundary_value, BC_Array):
    dim = 41.
    l = dim/6.0
    gbar = 1.0
    coordinate_value = boundary_value + 1
    if BC_Array[0] == 1 and (dim/4 - l/2) < coordinate_value < (dim/4 + l/2):
        t = coordinate_value - dim/4
        return gbar * (1 - (2*t/l)**2)
    if BC_Array[1] == 1 and ((3 * dim)/4 - l/2) < coordinate_value < ((3 * dim)/4 + l/2):
        t = coordinate_value - (3 * dim)/4
        return gbar * (1 - (2*t/l)**2) 
    return 0

def generateBCArrayNumpy(BCFunction, inputArray, outputArray):
    in_BC_Array = np.zeros(41) # using 41x41 numpy arrays
    out_BC_Array = np.zeros(41)
    # print(in_BC_Array)
    for i in range(41):
        in_BC_Array[i] = BCFunction(i, inputArray)
        out_BC_Array[i] = BCFunction(i, outputArray)
    return np.array([in_BC_Array, out_BC_Array])

def generateRandomInputs():
    result = []
    values = getPermutedListOfSix()
    values2 = getPermutedListOfSix()
    for i in range(len(values)):
        for j in range(len(values2)):
            result.append((generateBCArrayNumpy(parabolicFlowProfileBC, values[i], values2[j]), values[i], values2[i]))
        # break
        print(values[i])
    return result

inputs = generateRandomInputs()


# bc = generateBCArrayNumpy(parabolicFlowProfileBC, [1, 0], [0, 1])

outPossibilities = [[0, 1], [1, 0], [1, 1]]
inPossibilities = [[0, 1], [1, 0], [1, 1]]

for inBC in inPossibilities:
    for outBC in outPossibilities:
        bc = generateBCArrayNumpy(parabolicFlowProfileBC, inBC, outBC)
        for volfrac in range(10, 50):
            start_time = time.perf_counter()
            modelInput = torch.tensor(np.array([[bc[0]], [bc[1]], np.full((1, 41), volfrac)])).to(device).to(torch.float32)
            # print(modelInput.shape)
            output = model(modelInput)
            output = output.cpu().detach().numpy()
            output = np.reshape(output, (41, 41))
            elapsed_time = time.perf_counter() - start_time

            # print(f'Elapsed time (s): {elapsed_time}')
            with open(f'./dataforpaper/modelexectime/modeltest_in-{inBC[0]}-{inBC[1]}_out-{outBC[0]}-{outBC[1]}_{volfrac}.txt', 'w+') as f:
                f.write(f'Elapsed time (s): {elapsed_time}')
            
            plt.imsave(f'./modelevaloutputs/modeltest_in-{inBC[0]}-{inBC[1]}_out-{outBC[0]}-{outBC[1]}_{volfrac}.png', output, cmap='gray')
            
            with open(f'./modelnumpyoutputs/modeltest_in-{inBC[0]}-{inBC[1]}_out-{outBC[0]}-{outBC[1]}_{volfrac}.npy', 'wb+') as f:
                np.save(f, output)


# volfrac = 22
# for i in inputs:
#     for volfrac in range(10, 50):
#         boundaryCondition = i[0]
#         in_barrier = i[1]
#         out_barrier = i[2]
#         InputResult = torch.tensor(np.array([[boundaryCondition[0]], [boundaryCondition[1]], np.full((1, 41), volfrac)])).to(device).to(torch.float32)
#         # modelInput = torch.cat((topologyLabels, volFrac), -1).to(device)
#         output = model(InputResult)
#         output = output.cpu().detach().numpy()
#         output = np.reshape(output, (41, 41))
#         plt.imshow(output)
#         print(i[1])
#         strVal = '_'.join([str(j) for j in i[1]]) + "-" + '_'.join([str(j) for j in i[2]])
#         plt.imsave(f'./modeloutputs/modeltest-{volfrac}-{strVal}.png', output)