#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np
import argparse
import json
from PIL import Image


parser = argparse.ArgumentParser(description='Deep learning - Train a new network on a data set')
parser.add_argument('data_dir', type=str,
                    help='path to parent data directory',default="./flowers/")
parser.add_argument('--save_dir', type=str,
                    help='path to directory to save the checkpoints',default="./checkpoint_2.pth")
parser.add_argument('--network', type=str,
                    help='pretrained model as vgg13',default="densenet121")
parser.add_argument('--lr', type=float,
                    help='learning rate',default=.01)
parser.add_argument('--hidden_units', type=int,
                    help='hidden units',default=512)
parser.add_argument('--epochs', type=int,
                    help='number of epochs',default=5)
parser.add_argument('--device', type=str,
                    help='cuda or cpu?',default="cpu")


args = parser.parse_args()
#print(args.accumulate(args.integers))
data_dir = args.data_dir
save_dir = args.save_dir
network = args.network
lr = args.lr
hidden_units = args.hidden_units
epochs = args.epochs
device = args.device

# Load the data
def data_transform(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(p=0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

    valid_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64) 
    
    return train_data, trainloader, validloader, testloader

# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Building and training the classifier

def build_model(device, network, hidden_units, lr):
    if network == 'densenet121':
        model = models.densenet121 (pretrained = True)
    else:
        model = models.vgg13 (pretrained = True)    
    #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    if network == 'vgg13':
        model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, 102),
                                     nn.LogSoftmax(dim=1))
    else:
        model.classifier = nn.Sequential(nn.Linear(1024, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, 102),
                                     nn.LogSoftmax(dim=1))
    
    
    return model

def train_model(epochs, data_dir, device, network, hidden_units, lr):
    steps = 0
    running_loss = 0
    accuracy = 0
    print_every = 50
    train_losses, test_losses = [], []
    model = build_model(device, network, hidden_units, lr)
    train_data, trainloader, validloader, testloader = data_transform(data_dir)
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    if torch.cuda.is_available() and device == 'cuda':
        model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);
    for epoch in range(epochs):            
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            loss = criterion(log_ps, labels)            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
                    # Calculate accuracy
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        else:
            print(f"Training loss: {running_loss/len(trainloader)}..."
                 f"Training accuracy: {accuracy/len(trainloader):.3f}")
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy1 = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy1 += torch.mean(equals.type(torch.FloatTensor)).item()
            #Print training and validation loss                   
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(validloader):.3f}.. "
                      f"Test accuracy1: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train
    return model           
              
def test_model(data_dir, device):
    test_loss = 0
    accuracy = 0
    model = build_model(device, network, hidden_units, lr)
    if torch.cuda.is_available() and device == 'cuda':
        model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);
    
    model.eval()
    train_data, trainloader, validloader, testloader = data_transform(data_dir)
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
        
            test_loss += batch_loss.item()
        
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
    print(#f"Epoch {epoch+1}/{epochs}.. "
    f"Train loss: {running_loss/print_every:.3f}.. "
    f"Test loss: {test_loss/len(testloader):.3f}.. "
    f"Test accuracy: {accuracy/len(testloader):.3f}")
    
def save_checkpoint(data_dir, save_dir, model, device, network, hidden_units, lr ):
    train_data, trainloader, validloader, testloader = data_transform(data_dir)
    model.class_to_idx = train_data.class_to_idsx
    model = build_model(device, network, hidden_units, lr)
    checkpoint = {'classifier': model.classifier,
                  'droupout':0.2,
                  'lr':lr,
                  'epochs':epochs,
                  'state_dict': model.state_dict(),
                  'class_to_idx':model.class_to_idx,
                  'optimizer' : optimizer.state_dict(),}
    
    torch.save(checkpoint, save_dir)
    
def main():
    data_t = data_transform(data_dir)
    build_model(device, network, hidden_units, lr)
    train_model(epochs, data_dir, device, network, hidden_units, lr)
    test_model(data_dir, device)
    save_checkpoint(data_dir, save_dir, model, device, network, hidden_units, lr )
    

    
if __name__== "__main__":
    main()
    
    


