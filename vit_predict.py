
def myhelp():
    print ("")
    print ("How to use Vision Transformer Image Classification...")
    print ("1. Prediction         : python vit.py image_folder p weight_filename")
    print ("")

import sys, os
if len(sys.argv) > 1:
    dataset_dir = sys.argv[1] # dataset_folder
    if sys.argv[1] == '?':
        myhelp()
        sys.exit(1)
else:
    myhelp()
    sys.exit(1)


base_directory = dataset_dir  
NUM_WORKERS = os.cpu_count()


import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
# import datetime

from torch import nn
from torchvision import transforms
from helper_functions import set_seeds
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_subfolders(directory_path):
    """Return a list of subfolder names in the given directory."""
    subfolders = []
    # Check if the directory exists
    if os.path.exists(directory_path):
        # List all files and folders in the directory
        for item in os.listdir(directory_path):
            # Join the directory path with the item name
            full_path = os.path.join(directory_path, item)
            # Check if the item is a directory
            if os.path.isdir(full_path):
                subfolders.append(item)
    else:
        print("Directory does not exist.")
    return subfolders

import pandas as pd
import glob

def list_images_in_folder_with_pandas(folder_path):
    images = []
    for extension in ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp']:
        images.extend(glob.glob(os.path.join(folder_path, '**', extension), recursive=True))
    
    # Creating a DataFrame
    df = pd.DataFrame(images, columns=['Image_Path'])
    return df

# folder_name = "path_to_your_folder"  # Replace with your folder path
# image_df = list_images_in_folder_with_pandas(folder_name)

from typing import List, Tuple
from PIL import Image

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device='cpu'):
    
    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ### 

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    print (f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")

def main():
    # user_epoch = myepoch
    # if len(sys.argv) > 2 :
    #     user_epoch = sys.argv[2]
    #     if not user_epoch.isnumeric():
    #         user_epoch = myepoch
    #     else:
    #         user_epoch = eval(sys.argv[2])
    
    ######################################################################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print ("Device is " + device)
    ######################################################################
    
    class_names = get_subfolders(dataset_dir + '/_train')
    print("Class names :")
    print(class_names)
    ######################################################################
    # 1. Get pretrained weights for ViT-Base
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 

    # 2. Setup a ViT model instance with pretrained weights
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

    # 3. Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    set_seeds()
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

    # Setup custom image path
    custom_image_path = 'P2-034.jpg'

    import requests

    # Import function to make predictions on images and plot them 
    # from going_modular.going_modular.predictions import pred_and_plot_image
    # Predict on custom image
    pred_and_plot_image(model=pretrained_vit,
                        image_path=custom_image_path,
                        class_names=class_names,
                        device=device)
    ######################################################################
    # Setup custom image path
    custom_image_path = 'R-019.jpg'
    # Predict on custom image
    pred_and_plot_image(model=pretrained_vit,
                        image_path=custom_image_path,
                        class_names=class_names,
                        device=device)
    
    image_df = list_images_in_folder_with_pandas(base_directory)
    print(image_df)

if __name__ == "__main__":
    
    main()
