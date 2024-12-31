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
# reader.SetFileName('controloutput/left3/right2/fraction60/control_iterations_final000051.vtu')
# reader.Update()

# ug = reader.GetOutput()
# points = ug.GetPoints()

fileListRaw = os.listdir('./output')

BASE_PATH = "./controloutput/"
BASE_LEFT = "left"
BASE_RIGHT = "right"

os.listdir('./finaloutputs')
# print(os.listdir('./controloutput/left1/right1/fraction60'))
# for leftpath in os.listdir('./finaloutputs'):
#     for rightpath in os.listdir(f'./controloutput/{leftpath}'):
#         for volfrac in os.listdir(f'./controloutput/{leftpath}/{rightpath}'):
#             fileList = os.listdir(f'./controloutput/{leftpath}/{rightpath}/{volfrac}')
#             fileList.sort(key=lambda file: int(file[-8:-5]) if file[-5:] == '.xdmf' else -1) # last three digits for iteration no.
#             lastIteration = fileList[-1]
#             reader.SetFileName(f'controloutput/{leftpath}/{rightpath}/{volfrac}/{lastIteration}')
#             reader.Update()
#             ug = reader.GetOutput()
#             data = vtk_to_numpy(ug.GetPointData().GetScalars()).reshape((31, 31))
#             with open(f'./cleanedoutputs/{leftpath}_{rightpath}_{volfrac}.npy', "wb+") as f:
#                 np.save(f, data)
for leftpath in os.listdir('./finaloutputs'):
    for rightpath in os.listdir(f'./finaloutputs/{leftpath}'):
        for volfrac in os.listdir(f'./finaloutputs/{leftpath}/{rightpath}'):
            if volfrac[-5:] == '.xdmf':
                reader.SetFileName(f'./finaloutputs/{leftpath}/{rightpath}/{volfrac}')
                reader.Update()
                ug = reader.GetOutput()
                data = vtk_to_numpy(ug.GetPointData().GetScalars()).reshape((31, 31))
                with open(f'./cleanedoutputs/{leftpath}_{rightpath}_{volfrac[:-5]}.npy', "wb+") as f:
                    np.save(f, data)



# for i in os.listdir('./controloutput/left1/right1/fraction60'):
#     print(int(i[-7:-4]))

# fileListUpdated = []

# for file in fileListRaw:
#     if "control_iterations_final" in file:
#         cleanedFileName = file[25:-4]
#         fileListUpdated.append(cleanedFileName)

# fileListUpdatedAndFiltered = {}

# for i in range(35, 100):
#     tmpList = []
#     for fileNo in fileListUpdated:
#         # print(fileNo)
#         fn = deepcopy(fileNo)
#         # print(fn)
#         if int(fn[0:2]) == i:
#             tmpList.append(int(fileNo))
#             # print(int(fileNo))
#         # tmpList.append
#     finalFileNo = max(tmpList)
#     # print(tmpList)
#     fileListUpdatedAndFiltered[i] = f"./output/control_iterations_final_{finalFileNo}.vtu"
#     # fileListUpdatedAndFiltered.append(f"./output/control_iterations_final_{finalFileNo}.vtu")

# for key in fileListUpdatedAndFiltered.keys():
#     currValue = fileListUpdatedAndFiltered[key]
    
# dataArrayList

# print(fileListUpdatedAndFiltered)


# print(vtk_to_numpy(points.GetData()))
# data = vtk_to_numpy(ug.GetPointData().GetScalars()).reshape((26, 26))

# plt.imshow(data, interpolation="nearest", origin="upper")
# plt.colorbar()
# plt.show()

# print(math.sqrt(len(data)))

# values = vtk_to_numpy(ug.GetPointData().GetScalars())