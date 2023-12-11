import sys
if len(sys.argv) > 1:
    dataset_dir = sys.argv[1] #"dataset_ebs" # Replace with your dataset path
    if sys.argv[1] == '?':
        print("python vit_ebs.py dataset_directory [weight_file]")
        sys.exit(1)
else:
    print ("Pls specific directory of the dataset")
    sys.exit(1)

train_ratio = 0.75  # For instance 0.75 = 75% training and 25% validation
base_directory = dataset_dir  

import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(base_dir, train_ratio):
    classes = os.listdir(base_dir)
    
    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        if not os.path.isdir(class_dir):
            continue

        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        train_imgs, val_imgs = train_test_split(images, train_size=train_ratio)

        # Create training and validation directories
        train_dir = os.path.join(base_dir, '_train', cls)
        val_dir = os.path.join(base_dir, '_valid', cls)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Copy images to respective directories
        for img in train_imgs:
            shutil.copy(img, train_dir)
        for img in val_imgs:
            shutil.copy(img, val_dir)

import matplotlib.pyplot as plt
import torch
import torchvision
import datetime

from torch import nn
from torchvision import transforms
from helper_functions import set_seeds
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

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

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS):

  # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
    class_names = train_data.classes

  # Turn images into data loaders
    train_dataloader = DataLoader(  train_data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True)
    test_dataloader = DataLoader(   test_data,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True)

    return train_dataloader, test_dataloader, class_names

def main():
    ######################################################################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device
    ######################################################################
    
    class_names = get_subfolders(dataset_dir + '/_train')
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
    # pretrained_vit # uncomment for model output 
    ######################################################################
    from torchinfo import summary

    # Print a summary using torchinfo (uncomment for actual output)
    summary(model=pretrained_vit, 
            input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    ######################################################################
    # Setup directory paths to train and test images
    train_dir = (dataset_dir + '/_train')
    test_dir = (dataset_dir + '/_valid')
    ######################################################################
    # Get automatic transforms from pretrained ViT weights
    pretrained_vit_transforms = pretrained_vit_weights.transforms()
    print(pretrained_vit_transforms)
    ######################################################################
    # Setup dataloaders
    train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(train_dir=train_dir,
                                                                                                test_dir=test_dir,
                                                                                                transform=pretrained_vit_transforms,
                                                                                                batch_size=32)
    ######################################################################

    if len(sys.argv) < 3 : # Not specific weight filename 
        from going_modular.going_modular import engine

        # Create optimizer and loss function
        optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train the classifier head of the pretrained ViT feature extractor model
        set_seeds()
        pretrained_vit_results = engine.train(model=pretrained_vit,
                                            train_dataloader=train_dataloader_pretrained,
                                            test_dataloader=test_dataloader_pretrained,
                                            optimizer=optimizer,
                                            loss_fn=loss_fn,
                                            epochs=1,
                                            device=device)
        
        ######################################################################
        # Plot the loss curves
        from helper_functions import plot_loss_curves

        plot_loss_curves(pretrained_vit_results) 
        ######################################################################

        # Save trainend weight

        # Get current datetime to use in filename
        now = datetime.datetime.now()
        filename = dataset_dir + "/" + now.strftime("%Y%m%d-%H%M%S") + "_model.pth"

        # Save the model weights to disk
        torch.save(pretrained_vit.state_dict(), filename)

    # Load the model weights 
    if len(sys.argv) > 2:
        filename = dataset_dir + "/" + sys.argv[2] # saved weight filename
    loaded_weights = torch.load(filename)

    # load the weights 
    pretrained_vit.load_state_dict(loaded_weights)

    # Setup custom image path
    custom_image_path = 'P2-034.jpg'

    import requests

    # Import function to make predictions on images and plot them 
    from going_modular.going_modular.predictions import pred_and_plot_image

    # Predict on custom image
    pred_and_plot_image(model=pretrained_vit,
                        image_path=custom_image_path,
                        class_names=class_names)
    ######################################################################
    # Setup custom image path
    custom_image_path = 'R-019.jpg'
    # Predict on custom image
    pred_and_plot_image(model=pretrained_vit,
                        image_path=custom_image_path,
                        class_names=class_names)

if __name__ == "__main__":
    if not os.path.exists(dataset_dir + '/_train'):
        print(f"Directory '{dataset_dir + '/_train'}' does not exist, Generating training/validation dataset...")
        split_dataset(base_directory, train_ratio)
        print("Done.")
    
    main()