import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import sys
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BATCH_SIZE = 100

# LOAD DATA HERE

topologyList = []
topologyLabels = []
BASE_OUTPUT_PATH = './neuralnetworkdata/'

TOPOLOGY_PATH = './neuralnetworkdata/TopologyOutput/'
BCOUTPUT_PATH = './neuralnetworkdata/BCOutput/'

topologies = os.listdir(TOPOLOGY_PATH)
boundaryConditions = os.listdir(BCOUTPUT_PATH)
# print(topologies)

preprocessedData = []

for path in range(len(topologies)):
    topologyPath = TOPOLOGY_PATH + topologies[path]
    bcPath = BCOUTPUT_PATH + boundaryConditions[path]
    topology = np.load(topologyPath)
    boundaryCondition = np.load(bcPath)
    topologyconverted = np.array(topology, dtype=np.float32).reshape(41, 41)
    # print(boundaryCondition.shape)
    volfrac = int(topologies[path].split("_")[-1][:-4])
    volumeFractionData = np.full((1, 41), volfrac)

    dataList = [np.array([topologyconverted]), boundaryCondition, volumeFractionData]

    preprocessedData.append(dataList)


class GeneratedTopologyDataset(Dataset):
    def __init__(self, topology_list, topology_labels, volume_fractions, transform=None, target_transform=None):
        self.topology_list = topology_list
        self.topology_labels = topology_labels
        self.transform = transform
        self.volume_fractions = volume_fractions
        self.target_transform = target_transform
        self.labels = torch.cat((self.topology_labels, self.volume_fractions), 1).to(torch.float32)

    def __len__(self):
        return len(self.topology_labels)

    def __getitem__(self, idx):
        # print(label)
        generatedTopology = self.topology_list[idx]
        topologyLabel = self.topology_labels[idx]
        volumeFractionLabel = self.volume_fractions[idx]
        label = self.labels[idx]
        # print(label)
        # print(label.shape)
        return generatedTopology, label


topologyList = torch.tensor(np.array([data[0] for data in preprocessedData]))
topologyLabels = torch.tensor(np.array([np.array(data[1]) for data in preprocessedData]))
volumeFraction = torch.tensor(np.array([data[2] for data in preprocessedData]))

BATCH_SIZE = 1

dataset = GeneratedTopologyDataset(topologyList, topologyLabels, volumeFraction)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

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

learning_rate = 0.001

NUM_EPOCHS = 200

model = NeuralNet(41, 1681, 1681).to(device)
criterion = nn.L1Loss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
losses = []
averagedTrainLosses = []
averagedTestLosses = []

# TRAIN MODEL
for epoch in tqdm(range((NUM_EPOCHS))):
    tmpTrainLosses = []
    tmpTestLosses = []
    for i, (topologies, labels) in enumerate(train_loader):
        topologies = topologies.to(device)
        labels = labels.to(device)
        # print(labels.shape)
        
        # Forward pass
        outputs = model(labels)
        # print(outputs.shape)
        # print(topologies.shape)
        loss = criterion(outputs.flatten(), topologies.flatten())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        currloss = loss.item()
        losses.append(currloss)
        tmpTrainLosses.append(currloss)
    averagedTrainLosses.append(np.mean(tmpTrainLosses))
    writer.add_scalar('Loss/train', np.mean(tmpTrainLosses), epoch)
    
    # find test loss
    for i, (topologies, labels) in enumerate(test_loader):
        topologies = topologies.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(labels)
        # print(outputs.shape)
        # print(topologies.shape)
        loss = criterion(outputs.flatten(), topologies.flatten())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        currloss = loss.item()
        losses.append(currloss)
        tmpTestLosses.append(currloss)
    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {np.mean(tmpTestLosses):.4f}')
    averagedTestLosses.append(np.mean(tmpTestLosses))
    writer.add_scalar('Loss/test', np.mean(tmpTestLosses), epoch)

# plot losses with matplotlib
# print(len(averagedLosses))

# writer.close()

torch.save(model.state_dict(), "model.pth")
plt.plot([i + 1 for i in range(len(averagedTestLosses))], averagedTestLosses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Test Loss During Training")
plt.savefig("testlosses.png")
plt.show()
print("min test loss", min(averagedTestLosses))
plt.plot([i + 1 for i in range(len(averagedTrainLosses))], averagedTrainLosses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss During Training")
print("min train loss", min(averagedTrainLosses))
plt.savefig("trainlosses.png")
plt.show()