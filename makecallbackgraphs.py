import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

modelExecTimeDir = './dataforpaper/modelexectime/'
regularExecTimeDir = './dataforpaper/regularexectime/'
feaExecTimeDir = './dataforpaper/feaexectime/'

iterationsfrompredictioncallbackdir = './dataforpaper/iterationsfrompredictioncallback/'
regularcallbackdir = './dataforpaper/regularcallback/'
fiveiterationcallbackdir = './dataforpaper/fiveiterationcallback/'

modelExecTimes = os.listdir(modelExecTimeDir)
regularExecTimes = os.listdir(regularExecTimeDir)
feaExecTimes = os.listdir(feaExecTimeDir)

iterationsfrompredictioncallback = os.listdir(iterationsfrompredictioncallbackdir)
regularcallback = os.listdir(regularcallbackdir)
fiveiterationcallback = os.listdir(fiveiterationcallbackdir)

def filterOutFiles(inflow_1, inflow_2, outflow_1, outflow_2, file):
    metaData = file.split('_')[1:]
    inflow = [int(i) for i in metaData[0].split('-')[1:]]
    outflow = [int(i) for i in metaData[1].split('-')[1:]]
    volfrac = int(metaData[2].split('.')[0])
    if inflow == [inflow_1, inflow_2] and outflow == [outflow_1, outflow_2]:
        return (file, volfrac)
    return None

def readFilteredOutFile(filepath):
    with open(filepath, 'r') as f:
        return float(f.readlines()[0].split(':')[-1])

BCPossibilities = [[0, 1], [1, 0], [1, 1]]

count1 = 0
count2 = 0
fig, axs = plt.subplots(3, 3)

for inflow in range(len(BCPossibilities)):
    for outflow in range(len(BCPossibilities)):
        feaCallbacks = [
            filterOutFiles(BCPossibilities[inflow][0], BCPossibilities[inflow][1], BCPossibilities[outflow][0], BCPossibilities[outflow][1], i) for i in iterationsfrompredictioncallback if filterOutFiles(BCPossibilities[inflow][0], BCPossibilities[inflow][1], BCPossibilities[outflow][0], BCPossibilities[outflow][1], i) is not None
        ]
        feaCallbacks = [(readFilteredOutFile(iterationsfrompredictioncallbackdir + i[0]), i[1]) for i in feaCallbacks]
        regularCallbacks = [
            filterOutFiles(BCPossibilities[inflow][0], BCPossibilities[inflow][1], BCPossibilities[outflow][0], BCPossibilities[outflow][1], i) for i in regularcallback if filterOutFiles(BCPossibilities[inflow][0], BCPossibilities[inflow][1], BCPossibilities[outflow][0], BCPossibilities[outflow][1], i) is not None
        ]
        regularCallbacks = [(readFilteredOutFile(regularcallbackdir + i[0]), i[1]) for i in regularCallbacks]
        fiveiterationCallbacks = [
            filterOutFiles(BCPossibilities[inflow][0], BCPossibilities[inflow][1], BCPossibilities[outflow][0], BCPossibilities[outflow][1], i) for i in fiveiterationcallback if filterOutFiles(BCPossibilities[inflow][0], BCPossibilities[inflow][1], BCPossibilities[outflow][0], BCPossibilities[outflow][1], i) is not None
        ]
        fiveiterationCallbacks = [(readFilteredOutFile(fiveiterationcallbackdir + i[0]), i[1]) for i in fiveiterationCallbacks]
        # print(modelExecTimesList)

        # nnPipelineModelExecTimeList = [(feaExecTimesList[i][0] + modelExecTimesList[i][0], modelExecTimesList[i][1]) for i in range(len(modelExecTimesList))]

        # plt.scatter([i[1] for i in feaCallbacks], [i[0] for i in feaCallbacks], label='NN Pipeline', s=6)
        # plt.scatter([i[1] for i in regularCallbacks], [i[0] for i in regularCallbacks], label='Regular', s=6)
        # plt.scatter([i[1] for i in fiveiterationCallbacks], [i[0] for i in fiveiterationCallbacks], label='10 Iterations', s=6)

        axs[inflow, outflow].scatter([i[1] for i in feaCallbacks], [i[0] for i in feaCallbacks], label='NN Pipeline', s=4)
        axs[inflow, outflow].scatter([i[1] for i in regularCallbacks], [i[0] for i in regularCallbacks], label='Regular', s=4)
        axs[inflow, outflow].scatter([i[1] for i in fiveiterationCallbacks], [i[0] for i in fiveiterationCallbacks], label='10 Iterations', s=4)
        # axs[count1, count2].set_xlabel('Volume Fraction')
        # axs[count1, count2].set_ylabel('Objective Function Value')
        axs[inflow, outflow].set_title(f'Inflow: {BCPossibilities[inflow]}, Outflow: {BCPossibilities[outflow]}', fontdict={'fontsize': 12})
        print([inflow, outflow])
plt.xlabel('Volume Fraction')
plt.ylabel('Objective Function Value')
        # plt.title(f'Inflow: {inflow}, Outflow: {outflow}')
        # plt.legend()
        # plt.show()
plt.legend()
plt.show()


# for inflow in BCPossibilities:
#     count2 = 0
#     for outflow in BCPossibilities:
#         # modelExecTimesList = [
#         #     filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) for i in modelExecTimes if filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) is not None
#         # ]
#         # modelExecTimesList = [(readFilteredOutFile(modelExecTimeDir + i[0]), i[1]) for i in modelExecTimesList]
#         # regularExecTimesList = [
#         #     filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) for i in regularExecTimes if filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) is not None
#         # ]
#         # regularExecTimesList = [(readFilteredOutFile(regularExecTimeDir + i[0]), i[1]) for i in regularExecTimesList]
#         # feaExecTimesList = [
#         #     filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) for i in feaExecTimes if filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) is not None
#         # ]
#         # feaExecTimesList = [(readFilteredOutFile(feaExecTimeDir + i[0]), i[1]) for i in feaExecTimesList]
#         feaCallbacks = [
#             filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) for i in iterationsfrompredictioncallback if filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) is not None
#         ]
#         feaCallbacks = [(readFilteredOutFile(iterationsfrompredictioncallbackdir + i[0]), i[1]) for i in feaCallbacks]
#         regularCallbacks = [
#             filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) for i in regularcallback if filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) is not None
#         ]
#         regularCallbacks = [(readFilteredOutFile(regularcallbackdir + i[0]), i[1]) for i in regularCallbacks]
#         fiveiterationCallbacks = [
#             filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) for i in fiveiterationcallback if filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) is not None
#         ]
#         fiveiterationCallbacks = [(readFilteredOutFile(fiveiterationcallbackdir + i[0]), i[1]) for i in fiveiterationCallbacks]
#         # print(modelExecTimesList)

#         # nnPipelineModelExecTimeList = [(feaExecTimesList[i][0] + modelExecTimesList[i][0], modelExecTimesList[i][1]) for i in range(len(modelExecTimesList))]

#         plt.scatter([i[1] for i in feaCallbacks], [i[0] for i in feaCallbacks], label='NN Pipeline', s=6)
#         plt.scatter([i[1] for i in regularCallbacks], [i[0] for i in regularCallbacks], label='Regular', s=6)
#         plt.scatter([i[1] for i in fiveiterationCallbacks], [i[0] for i in fiveiterationCallbacks], label='10 Iterations', s=6)

#         axs[count1, count2].scatter([i[1] for i in feaCallbacks], [i[0] for i in feaCallbacks], label='NN Pipeline', s=4)
#         axs[count1, count2].scatter([i[1] for i in regularCallbacks], [i[0] for i in regularCallbacks], label='Regular', s=4)
#         axs[count1, count2].scatter([i[1] for i in fiveiterationCallbacks], [i[0] for i in fiveiterationCallbacks], label='10 Iterations', s=4)
#         # axs[count1, count2].set_xlabel('Volume Fraction')
#         # axs[count1, count2].set_ylabel('Objective Function Value')
#         axs[count1, count2].set_title(f'Inflow: {inflow}, Outflow: {outflow}')
#         print([count1, count2])
#         count2 += 1
#         plt.xlabel('Volume Fraction')
#         plt.ylabel('Objective Function Value')
#         plt.title(f'Inflow: {inflow}, Outflow: {outflow}')
#         # plt.legend()
#         # plt.show()
#     count1 += 1
# plt.legend()
# plt.show()