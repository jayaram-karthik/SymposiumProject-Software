import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata
import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import os
from copy import deepcopy
import math

reader = vtk.vtkXdmfReader()
# reader.SetFileName('fourpipe/finaloutputs/input-1-1/output-0-1/control_solution_final_29.xdmf')
# reader.Update()

# ug = reader.GetOutput()
# # print(ug)
# points = ug.GetPointData()
# # print(points)
# pointdata = points.GetScalars()
# # print(pointdata)
# numpyArray = vtk_to_numpy(pointdata).reshape(41, 41)
# # print(numpyArray)
# plt.imshow(numpyArray)
# plt.show()
# fileListRaw = os.listdir('./output')

# BASE_PATH = "./controloutput/"
# BASE_LEFT = "left"
# BASE_RIGHT = "right"

# os.listdir('./finaloutputs')
DATA_ROOT = "./fourpipe/"
BASE_PATH = DATA_ROOT + "finaloutputs/"

OUTPUT_ROOT = './neuralnetworkdata/'
# print(os.listdir('./controloutput/left1/right1/fraction60'))

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
    in_BC_Array = np.zeros(41) # using 40x40 numpy arrays
    out_BC_Array = np.zeros(41)
    # print(in_BC_Array)
    for i in range(41):
        in_BC_Array[i] = BCFunction(i, inputArray)
        out_BC_Array[i] = BCFunction(i, outputArray)
    return in_BC_Array, out_BC_Array

# in_BC_test, out_BC_test = generateBCArrayNumpy(parabolicFlowProfileBC, [1, 1], [1, 1])
# print(in_BC_test)
# print(out_BC_test)
            


for leftpath in os.listdir(BASE_PATH):
    LPath = BASE_PATH + f'{leftpath}/'
    for rightpath in os.listdir(LPath):
        RPath = LPath + f'{rightpath}/'
        for volfrac in os.listdir(RPath):
            if (volfrac.split('.')[-1] != 'xdmf'):
                continue
            VPath = RPath + f'{volfrac}'
            # print(volfrac.split('.')[0].split('_')[-1])
            volumeFraction = int(volfrac.split('.')[0].split('_')[-1])
            leftBC = [int(i) for i in leftpath.split('-')[1:]]
            rightBC = [int(i) for i in rightpath.split('-')[1:]]

            leftBC_label, rightBC_label = generateBCArrayNumpy(parabolicFlowProfileBC, leftBC, rightBC)

            FullBC = np.array([leftBC_label, rightBC_label])

            BCPath = f'{OUTPUT_ROOT}/BCOutput/{leftpath}_{rightpath}_{volumeFraction}.npy'

            with open(BCPath, "wb+") as f:
                np.save(f, FullBC)

            reader.SetFileName(VPath)
            reader.Update()
            ug = reader.GetOutput()

            points = ug.GetPointData()
            pointdata = points.GetScalars()


            data = vtk_to_numpy(pointdata).reshape((41, 41))

            dataPath = f'{OUTPUT_ROOT}/TopologyOutput/{leftpath}_{rightpath}_{volumeFraction}.npy'

            with open(dataPath, "wb+") as f:
                np.save(f, data)