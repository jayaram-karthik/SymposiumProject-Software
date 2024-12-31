import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

modelExecTimeDir = './dataforpaper/modelexectime/'
regularExecTimeDir = './dataforpaper/regularexectime/'
feaExecTimeDir = './dataforpaper/feaexectime/'

iterationsfrompredictioncallbackdir = './dataforpaper/iterationsfrompredictioncallback/'
regularcallbackdir = './dataforpaper/regularcallback/'

modelExecTimes = os.listdir(modelExecTimeDir)
regularExecTimes = os.listdir(regularExecTimeDir)
feaExecTimes = os.listdir(feaExecTimeDir)

iterationsfrompredictioncallback = os.listdir(iterationsfrompredictioncallbackdir)
regularcallback = os.listdir(regularcallbackdir)

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

count = 0
fig, axs = plt.subplots(3, 3)

count1 = 0
count2 = 0

for inflow in BCPossibilities:
    count2 = 0
    for outflow in BCPossibilities:
        modelExecTimesList = [
            filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) for i in modelExecTimes if filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) is not None
        ]
        modelExecTimesList = [(readFilteredOutFile(modelExecTimeDir + i[0]), i[1]) for i in modelExecTimesList]
        regularExecTimesList = [
            filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) for i in regularExecTimes if filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) is not None
        ]
        regularExecTimesList = [(readFilteredOutFile(regularExecTimeDir + i[0]), i[1]) for i in regularExecTimesList]
        feaExecTimesList = [
            filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) for i in feaExecTimes if filterOutFiles(inflow[0], inflow[1], outflow[0], outflow[1], i) is not None
        ]
        feaExecTimesList = [(readFilteredOutFile(feaExecTimeDir + i[0]), i[1]) for i in feaExecTimesList]
        # print(modelExecTimesList)

        nnPipelineModelExecTimeList = [(feaExecTimesList[i][0] + modelExecTimesList[i][0], modelExecTimesList[i][1]) for i in range(len(modelExecTimesList))]

        # plt.plot([i[1] for i in nnPipelineModelExecTimeList], [i[0] for i in nnPipelineModelExecTimeList], label='NN Pipeline')
        # plt.plot([i[1] for i in regularExecTimesList], [i[0] for i in regularExecTimesList], label='Regular')
        # plt.xlabel('Volume Fraction')
        # plt.ylabel('Execution Time (s)')
        # plt.title(f'Inflow: {inflow}, Outflow: {outflow}')
        # plt.legend()
        # plt.show()
        # plt.scatter([i[1] for i in nnPipelineModelExecTimeList], [i[0] for i in nnPipelineModelExecTimeList], label='NN Pipeline', s=6)
        # plt.scatter([i[1] for i in regularExecTimesList], [i[0] for i in regularExecTimesList], label='Regular', s=6)

        # axs[count1, count2].set_xlabel('Volume Fraction')
        # axs[count1, count2].set_ylabel('Execution Time (s)')
        axs[count1, count2].scatter([i[1] for i in nnPipelineModelExecTimeList], [i[0] for i in nnPipelineModelExecTimeList], label='NN Pipeline', s=6)
        axs[count1, count2].scatter([i[1] for i in regularExecTimesList], [i[0] for i in regularExecTimesList], label='Regular', s=6)
        axs[count1, count2].set_title(f'Inflow: {inflow}, Outflow: {outflow}', fontdict={'fontsize': 12})
        count2 += 1
        # print(nnPipelineModelExecTimeList)
        # count += 1
        # print(count)
    count1 += 1
plt.xlabel('Volume Fraction')
plt.ylabel('Execution Time (s)')
plt.legend()
plt.show()



# execTimes = [
#         filterOutFiles(1, 0, 0, 1, i) for i in modelExecTimes if filterOutFiles(1, 0, 0, 1, i) is not None
#     ]

# execTimes = [(readFilteredOutFile(modelExecTimeDir + i[0]), i[1]) for i in execTimes]
# print(execTimes)
