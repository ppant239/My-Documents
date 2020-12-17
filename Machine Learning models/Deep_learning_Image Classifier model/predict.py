import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import argparse
import json
from PIL import Image
import train
import seaborn as sns


parser = argparse.ArgumentParser(description='Deep learning - Train a new network on a data set')
parser.add_argument('image_path', type=str,
                    help='path to image',default="./flowers/train/1/image_06740.jpg")
parser.add_argument('--top_k', type=int,
                    help='top k predictions',default=5)
parser.add_argument('--category_names', type=str,
                    help='category name mapping',default="cat_to_name.json")
parser.add_argument('--device', type=str,
                    help='cuda or cpu?',default="cpu")

# for testing
parser.add_argument('--checkpoint', type=str,
                    help='checkpoint path from part 1 (for testing)',default="checkpoint.pth")


args = parser.parse_args()
#print(args.accumulate(args.integers))

#filepath = train.args.save_dir
filepath = args.checkpoint
image_path = args.image_path
top_k = args.top_k
device = args.device
category_names = args.category_names

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image_path):
 # TODO: Process a PIL image for use in a PyTorch model

    im = Image.open(image_path)
    width, height = im.size   # Get dimensions
    
 # Resize the image
    if(width < 256 or height < 256 ):
        im = im.thumbnail(256)
    
 # crop from center

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    
    pil_image = im.crop ((left, top, right, bottom))
    
    np_image = np.array(pil_image)/255
    #Ref: https://stackoverflow.com/questions/44955656/how-to-convert-rgb-pil-image-to-numpy-array-with-3-channels
 # Normalize   
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image-mean)/std 
 # Transpose
    np_image = np_image.transpose((2, 1, 0))
    
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, filepath, top_k, device, category_names):
    #load the model
    model = load_checkpoint(filepath) # model = 'checkpoint.pth'
    #process image
    test_image = process_image(image_path)
    #display_image = imshow(test_image)
    #print('test image', display_image)
    
    img_tensor = torch.from_numpy(test_image).type(torch.FloatTensor)
    image = img_tensor.unsqueeze(0)
    
    #image_u = Variable(image)
    image_u = image.to(device)

    model.eval()
    with torch.no_grad ():
        image_u = model.forward(image_u)
    # TODO: Calculate the class probabilities (softmax) for img
    ps = torch.exp(image_u)
    
    #top_p, top_class = np.array(ps.topk(1, dim=1))   
    top_p, top_class = torch.topk(ps, topk)
    top_p_array = np.array(top_p)[0]
    top_c_array = np.array(top_class)[0]
    
    
    class_to_idx = model.class_to_idx
    indx_to_class = {x: y for y, x in class_to_idx.items()}
    classes = [indx_to_class[i] for i in top_c_array]
    
    return top_p_array, classes

def main():
    model = load_checkpoint(filepath)
    print(model)
    sns.set_style("darkgrid")
    image = process_image(image_path)
    probs,classes = predict(image_path, filepath, topk=5)
    
    print("Probability {:.2f}..%:".format(probs))
    print("Class names:", classes)
    
    names = []
    for i in classes:
        names += [category_names[i]]
    print(names)
    
    #fig = plt.figure()
    #imshow(image)
    #plt.title(names[0])
    #plt.show()

    #sns.barplot(x=probs, y=names,label="Total", color= 'b');

    
if __name__== "__main__":
    main()

