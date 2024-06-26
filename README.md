# Custom 3D Convolutional Neural Network (CNN)

This is a custom 3D Convolutional Neural Network (CNN) model for 3D object detection made using PyTorch, which has an accuracy of approximately 85%.

**Note:** This code was written in Google Colab. Therefore, I have used the Nvidia T4 GPU and 12 GB RAM provided by Google Colab.

## Dataset:
The dataset I used for this 3D CNN is the [ModelNet10](https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset) dataset from Kaggle. This dataset contains 4,899 3D CAD models divided into 10 different categories. I split this dataset according to the train-test split ratio recommended by the actual authors of this dataset: 3,991 (80%) models for training and 908 (20%) models for testing. To use this dataset, I downloaded it and stored it in my Google Drive.

## Data Preprocessing

### Voxelization of ModelNet10:
Since a 3D CNN does not directly take mesh objects (3D objects) as input, I performed voxelization on this dataset by converting all the mesh objects present in this dataset to voxel grids and then subsequently into voxel grid arrays. These arrays were stored in my Google Drive in a folder named "ModelNet10_arrays". The structure of this folder is similar to the structure of the ModelNet10 dataset folder, but the only difference is that this folder contains the voxel grid arrays in .npz files instead of mesh objects in .off files. (The code for this voxelization process is present in my [Voxelization](https://github.com/MohammadKatif/Voxelization/tree/main) repository).

### Dataset and Dataloader:
After performing voxelization, to use the voxelized data, I loaded the "ModelNet10_array" from my Google Drive and created [custom PyTorch Datasets and Dataloaders](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) using it. Specifically, I created a training dataset, a training dataloader, a testing dataset, and a testing dataloader to use the ModelNet10_array data in my 3D CNN.

## Model Architecture:
![image](https://github.com/MohammadKatif/Custom-3D-CNN/assets/143898427/ea71463a-7c40-4a63-a928-1f04e1b566f8)

### Detailed Explanation:
The dimension of the voxel grid arrays stored in the dataset is 100x100x100. Unlike 2D CNNs, this input dimension is really large for a 3D CNN, which increases the computational demand and training time. To address this issue, the first layer I defined in the model is a 3D max pooling layer with a kernel size and stride of (2x2x2). This reduces the dimensions of the input voxel grid from 100x100x100 to 50x50x50 while retaining the most important information. After reducing the size of the input by half, I passed the input through two 3D convolutional layers, both with a kernel size of 3x3x3, stride size of 1x1x1, and padding size of 1x1x1. Since the input data is in grayscale, the first 3D convolutional layer takes 1 channel and outputs 16 channels, extracting basic features from the input. The second 3D convolutional layer takes 16 channels and outputs 32 channels, hence extracting more advanced features from the input. Then, the output of these convolutional layers goes through the same 3D max pooling layer, reducing its size to (25x25x25). Lastly, the output from the pooling layer passes through a flattening layer and then through two fully connected linear layers. The input size of the first fully connected layer is 500,000 and its output size is 128, while the input size of the second fully connected layer is 128 and its output size is 10 (because ModelNet10 contains 10 different categories).

## Training and Testing:
This model was trained for only 4 epochs, with the training data containing the voxel grids of 3,991 models, which is 80% of the ModelNet10 dataset. After training, the model was tested on the entire testing data containing voxel grids of 908 models, which is 20% of the ModelNet10 dataset. The accuracy of the model was calculated by determining the percentage of correctly predicted labels in the entire testing dataset.

## Accuracy:
The accuracy of the model was calculated by determining the percentage of correctly predicted labels in the entire testing dataset. Currently, the accuracy of this model is exactly **85.79295154185021%**.
