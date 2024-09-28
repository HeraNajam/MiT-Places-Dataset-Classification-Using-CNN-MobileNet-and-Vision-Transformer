# MiT-Places-Dataset-Classification-Using-CNN-MobileNet-and-Vision-Transformer

Introduction:
This project is focused on a classification task involving over 1 million images categorized into 365 distinct classes. To achieve this, three different deep learning models were utilized:
1.	Convolutional Neural Network (CNN)
2.	MobileNet with pre-trained ImageNet weights
3.	Vision Transformer (ViT)
    
This README will guide you through the project's setup, usage, and the detailed workings of these models.

Dataset and Preprocessing:
This project uses the MIT Places dataset, sourced from Kaggle at https://www.kaggle.com/datasets/nickj26/places2-mit-dataset. The dataset contains a total of 1,803,460 images, categorized into 365 classes.
The dataset is organized into folders, with each folder representing a class for the images it contains.
Initially, the images were sized at 250 x 250 pixels. For more efficient model execution, they were resized to 120 x 120 pixels.
The "places-classification-data-preprocessing" notebook shows the data preprocessing steps and creates a CSV file with columns for file name and class.

Modeling:
This project used Kaggle for training the models and saved the best-performing models.
Train, Test, and Validation Split
1. Train set size: 1,082,076 (60%)
2. Test set size: 360,692 (20%)
3. Validation set size: 360,692 (20%)

Configurations for Training
1. Platform: Kaggle
2. Device: GPU
3. Framework: PyTorch

Training Parameters
1. Loss Function: CrossEntropyLoss
2. Optimizer: Adam
3. Learning Rate: 0.0001
4. Batch Size: 1000
5. Training Epochs: 20

1 - Classification Using CNN
The "places-classification-128-cnn" notebook details the training process with a CNN model. It used a total of 23,278,765 parameters and saved the model with the lowest loss value.

2 - Classification Using MobileNet
The "places-classification-128-mobilenet" notebook outlines the training process with a MobileNet model. It utilized pre-trained weights from IMAGENET1K_V1 and trained with a total of 3,277,677 parameters. The model with the lowest loss value was saved.

3 - Classification Using ViT
The "places-classification-128-ViT" notebook explains the training process with a Vision Transformer (ViT) model. It used a total of 1,108,205 parameters and saved the model with the lowest loss value.

ViT Parameters
1. Patch Size:
2. Input Channels: 3
3. Embedding Dimension: 192
4. Number of Heads: 3
5. Number of Blocks: 2
6. Dropout: 0.1

Each notebook took approximately 12 hours to train, and the best models were saved. These notebooks can be found in the folder named "pre-trained models."


Dashboard:
The main.py file contains the main code for the dashboard, which uses the Streamlit library. The dashboard has four main parts:
1.	Display Images: This section includes drop-down menus allowing users to select the range of images to display, such as 1-50, 51-100, 101-150, etc.

2.	Test a Sample Image using Three Models: This section uses the cnn_test.py, mobnet_test.py, and vit_test.py files to load the saved best models. These files take an input image and classify it using the CNN, MobileNet, and ViT models, respectively.

3.	Display the Results for Individual Models: This section uses the display_metrics_pre_trained_models.py file to show the loss and accuracy graphs for individual models based on training and validation data.

4.	Display the Combined Results of All Models: This section also uses the display_metrics_pre_trained_models.py file to display the loss and accuracy graphs for the combined models based on training and validation data.

