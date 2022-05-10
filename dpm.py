import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
from torchvision.models import alexnet, vgg16
from basicNet import basicnet
from scipy.spatial.distance import cosine
import json
import time
import pickle
import argparse
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
######################################
####### BEGIN HELPER FUNCTIONS #######
######################################

def train(train_dataloader, model, model_params):
    model.train()

    total_train_loss = 0
    total_train_correct = 0
    criterion = model_params['criterion']
    optimizer = model_params['optimizer']
    for (inputs, targets) in tqdm(train_dataloader, leave=False):
        # Zero the Gradients
        optimizer.zero_grad()
        
        # Get Predictions
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        # Loss Per Batch
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        
        # Correct Per Batch
        _, predicted = outputs.max(1)
        total_train_correct += predicted.eq(targets).sum().item()
    
    mean_train_loss = total_train_loss / model_params['TRAIN_SIZE']
    train_accuracy = total_train_correct / model_params['TRAIN_SIZE']
    
    return mean_train_loss, train_accuracy
            
def test(test_dataloader, model, model_params, isVal=False):                 
    model.eval()
    total_test_loss = 0
    total_test_correct = 0
    criterion = model_params['criterion']

    with torch.no_grad():
        for (inputs, targets) in test_dataloader:
            
            # Get predictions
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            
            # Loss Per Batch
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()
            
            # Correct Per Batch
            _, predicted = outputs.max(1)
            total_test_correct += predicted.eq(targets).sum().item()
        
    if isVal:
        mean_test_loss = total_test_loss / model_params['VAL_SIZE']
        test_accuracy = total_test_correct / model_params['VAL_SIZE']
    else:
        mean_test_loss = total_test_loss / model_params['TEST_SIZE']
        test_accuracy = total_test_correct / model_params['TEST_SIZE']
        
    
    return mean_test_loss, test_accuracy


def run_simulations(train_dataloader, val_dataloader, model, model_params):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    times = []
    
    pbar = tqdm(range(model_params['EPOCHS']))
    
    for i in pbar:
        # Train
        start = time.time()
        mean_train_loss, train_accuracy = train(train_dataloader, model, model_params)
        end = time.time()
        
        # Validation
        mean_val_loss, val_accuracy = test(val_dataloader, model, model_params, isVal=True)
        
        # Get Losses Per Epoch
        train_losses.append(mean_train_loss)
        val_losses.append(mean_val_loss)
        
        # Get Accuracies Per Epoch
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Get Modeling Times Per Epoch
        times.append(end - start)
        
#         # Scheduler
#         scheduler.step()
        
        pbar.set_postfix({'train_loss': mean_train_loss, 'train_accuracy': train_accuracy,
                          'val_loss': mean_val_loss, 'val_accuracy': val_accuracy})
        
        # Convergence/Early Stopping Criterion
        if i > 1:
            if np.abs(val_losses[-1] - val_losses[-2]) < 0.000001:
                break
        
        PATH = f'models/{model.__class__.__name__}_Epoch_{i}.pth'
        torch.save(model.state_dict(), PATH)
        
    # Test
    # test_loss, test_accuracy = test(test_dataloader, model, model_params)    
    
    
    return train_losses, val_losses

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def normalize(v):
    # v : numpy vector
    v = v - v.min()
    v = v / v.max()
    return v

def get_sim_judgments(images1,images2, model, layer='fc'):
    # images1 : list of N images (each image is a tensor)
    # images2 : list of N images (each image is a tensor)
    # layer : which layer do we want to use? fully connected ('fc'), 
    #         first convolutional ('conv1'), or second convolutional ('conv2')
    #
    # Return
    #  v_sim : N-dimensional vector
    # model.load_state_dict(torch.load(path_to_model))
    model.eval()
    N = images1.size()[0] # number of pairs
    assert N == images2.size()[0]
    
    sim_model = model
    sim_model.classifier[-1] = Identity()
    output1 = sim_model(images1)
    output2 = sim_model(images2)
    
    # flatten the tensors for each image
    T1 = output1.detach().cpu().numpy().reshape(N,-1)
    T2 = output2.detach().cpu().numpy().reshape(N,-1)

    v_sim = np.zeros(N)
    for i in range(N): # for each pair
        v1 = T1[i,:]
        v2 = T2[i,:]
        v_sim[i] = 1-cosine(v1,v2) # using cosine distance 
    
    return v_sim

######################################
######## END HELPER FUNCTIONS ########
######################################

# Data Normalization/Transformation
print('Creating dataset')
data_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Obtaining data from directory
dataset = ImageFolder('wikiart_4', transform=data_tf)
test_dataset = ImageFolder('style_test_wikiart/', transform=data_tf)
test_pairs = ImageFolder('wiki_test/', transform=data_tf)

# TRAIN - VAL SPLIT
n = 8000
print(f'Separating into train/val with {n} images from original dataset')
random_ids = np.random.choice(len(dataset), n, replace=False)
train_dataset = torch.utils.data.Subset(dataset, random_ids[:n-n//4])
val_dataset = torch.utils.data.Subset(dataset, random_ids[n-n//4:])

# PARAMETERS
model_params = {
    'EPOCHS': 100,
    'TRAIN_SIZE': len(train_dataset),
    'VAL_SIZE': len(val_dataset),
    'TEST_SIZE': len(test_dataset),
    'optimizer': None,
    'criterion': None
}

# Load Data to Dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=96, shuffle=False, num_workers=2)

print('Creating Models')
N_CLASSES = 4
models = [
    basicnet(num_classes=N_CLASSES).to(device),
    alexnet(num_classes=N_CLASSES).to(device),
    vgg16(num_classes=N_CLASSES).to(device),
]


images1 = [test_pairs[i][0] for i in range(48)]
images2 = [test_pairs[i][0] for i in range(48, 96)]

images1 = torch.stack(images1).to(device)
images2 = torch.stack(images2).to(device)

model_dict_style = {}
similarities = {}

for model in models:
    name = model.__class__.__name__
    print(f'Model: {name}')
    
    model_params['criterion'] = nn.CrossEntropyLoss()
    model_params['optimizer'] = optim.SGD(model.parameters(), lr=0.01)
    epochs = model_params['EPOCHS']
    
    print(f'Training on {epochs} epochs')
    train_losses, val_losses = run_simulations(train_dataloader,
                                            val_dataloader,
                                            model,                                           
                                            model_params)

    _, test_accuracy_style = test(test_dataloader, model, model_params)
    
    v_sim = get_sim_judgments(images1, images2, model)
    
    
    print(f'Saving {name} to file')
    model_dict_style[name] = [train_losses, val_losses, test_accuracy_style]
    similarities[name] = v_sim
    
    with open(f'model_dict_style.pkl', 'wb') as f:
        pickle.dump(model_dict_style, f)

    with open('similarities.pkl', 'wb') as f:
        pickle.dump(similarities, f)