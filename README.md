# Custom Image Classification Using Vision transformer
Image Classification Using Vision transformer with pretrained weight

### Steps to follow:
#### Installation:
Anaconda Python Environment <br/>
version is working for CPU or (GPU) <br/>
Python 3.8 <br/>
torchvision 0.16.1 (pip3 install torchvision==0.16.1) <br/>
torch 2.1.1 or (torch 2.1.1+cu121) (https://pytorch.org/get-started/locally/) <br/>
vit-pytorch (pip3 install vit-pytorch==1.6.4) <br/>
scikit-learn <br/>
mathplotlib <br/>

#### Setup:
1. Unzip "dataset_ebs.zip" (for your project it is your dataset zip file with subfolder is the image class name)<br/>
2. Run "dataset_spliter.ipynb" with editing dataset folder name and define the % of training and validation in the 1st cell prior run <br/>
3. Run "vit_ebs.ipynb" with editing dataset directory in the 3th cell prior run for the model training, model weight saving and prediction sample images<br/>
#### Reference: 
https://github.com/AarohiSingla/Image-Classification-Using-Vision-transformer <br/>
https://youtu.be/awyWND506NY <br/>

