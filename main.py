import os
import json
import pandas as pd
import numpy as np
import streamlit as st

from PIL import Image

from mobnet_test import mobnet_predict_function
from cnn_test import cnn_predict_function
from vit_test import vit_predict_function
from display_metrics_pre_trained_models import display_function

# Function to load images within a specified range based on numeric prefix
def load_images(image_dir, start_idx, end_idx):
    
    # List all image files
    image_files = [file for file in os.listdir(image_dir) if file.endswith(('png', 'jpg', 'jpeg'))]
    
    # Sort image files based on numeric prefix
    image_files.sort(key=lambda x: int(x.split('_')[0]))
    
    # Filter images within the specified range
    filtered_images = [os.path.join(image_dir, file) for file in image_files if start_idx <= int(file.split('_')[0]) < end_idx]
    
    return filtered_images

# Function to resize image
def resize_image(image, width=None, height=None):
    # Get the original dimensions
    original_width, original_height = image.size
    
    # If both width and height are specified
    if width and height:
        return image.resize((width, height))

########################################

# Paths to directories
results_dir = 'pre-trained-results'
image_dir = 'unique_images'
test_images = 'test_images'

cnn_history_path = 'pre-trained-results/CNN/CNN_2_history.json'
mob_history_path = 'pre-trained-results/Mobilenet/mobilenet_CNN_history.json'
vit_history_path = 'pre-trained-results/ViT/ViT_history_updated.json'

########################################

# Initialize session state
if 'option' not in st.session_state:
    st.session_state.option = 'Select a range'
if 'option2' not in st.session_state:
    st.session_state.option2 = 'Select from below'
if 'train_model' not in st.session_state:
    st.session_state.train_model = 'Select a model to train'
    
def reset_selectboxes(exclude):
    if exclude != 'option':
        st.session_state.option = 'Select a range'
    if exclude != 'option2':
        st.session_state.option2 = 'Select from below'
    if exclude != 'train_model':
        st.session_state.train_model = 'Select a model to train'

# Display initial image and title
if (st.session_state.option == 'Select a range' and 
    st.session_state.option2 == 'Select from below' ):
    
    st.title("MIT Places Classification Model Testing and Metrics Display ")
    initial_image_path = "display_image.png"  # Replace with your image path
    initial_image = Image.open(initial_image_path)
    st.image(initial_image, caption=" ", use_column_width=True)


# Sidebar for selecting range
with st.sidebar:
    
    # Drop down menu - 1 for selecting image range
    st.header('Display images - Mit Places Dataset')
    option = st.selectbox(
        ' ',
        ('Select a range','0-50', '50-100', '100-150', '150-200', '200-250', '250-300','300-350' ),
        key = 'option',
        on_change = lambda: reset_selectboxes('option')
    )
    
    # Drop down menu - 2 for selecting image test or metrics
    st.header('Use pre-trained models to test an Image/ Display pre-trained model metrics ')  
    option2 = st.selectbox(
        ' ',
        ( 'Select from below' ,'Select an image to test', 'Display individual metrics', 'Display combined metrics' ),
        key = 'option2',
        on_change = lambda: reset_selectboxes('option2')
    )
    
# =============================================================================
#     # Drop down menu - 3 for selecting model to train
#     st.header('Train a new model')  
#     train_model = st.selectbox(
#         ' ',
#         ( 'Select a model to train', 'CNN', 'Mobilenet', 'ViT'),
#         key = 'train_model',
#         on_change = lambda: reset_selectboxes('train_model')
#     )    
# =============================================================================

###########################################################################
# Determine the start and end indices based on the selection
if option != 'Select a range' and option2 == 'Select from below':
    
    if option == '0-50':
        start_idx, end_idx = 0, 50
    elif option == '50-100':
        start_idx, end_idx = 50, 100
    elif option == '100-150':
        start_idx, end_idx = 100, 150
    elif option == '150-200':
        start_idx, end_idx = 150, 200
    elif option == '200-250':
        start_idx, end_idx = 200, 250
    elif option == '250-300':
        start_idx, end_idx = 250, 300
    elif option == '300-350':
        start_idx, end_idx = 300, 350
    
    
    # Load images
    images = load_images(image_dir, start_idx, end_idx)
    
    # Display the images in the main area
    st.title('Images with label ' + option)
    
    # Define the number of columns
    num_columns = 5
    
    # Create a grid to display images
    cols = st.columns(num_columns)
    
    for idx, image_path in enumerate(images):
        col_idx = idx % num_columns
        with cols[col_idx]:
            # Display image name
            image_name = os.path.basename(image_path)
            st.text(image_name)
            # Display image
            img = Image.open(image_path)
            st.image(img, use_column_width=True)
    
    if len(images) == 0:
        st.write("No images found in the selected range.")

###########################################################################

elif option2 != 'Select from below':
    
    if option2 == 'Select an image to test':
    
        st.title('Upload a Test Image')
        uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            
            # Display the uploaded image
            img = Image.open(uploaded_file)
            
            resized_image = resize_image(img, width=400, height=400)
            st.image( resized_image)
            
            # Display the name of the uploaded image
            image_name = uploaded_file.name
            print(f"Name of the uploaded image: {image_name}")
            
            # Display the label
            label = int(image_name.split('t')[1].split('_')[1])
            print(f"Label: {label}")
            
            # Display the shape of the uploaded image
            print(f"Shape of the uploaded image: {img.size} (width, height)")
            
            st.write('PREDICTIONS WITH THE FOLLOWING MODELS')
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ### Model ...... CNN
                cnn_model_path = 'pre-trained-results/CNN/_best.pth'
                cnn_pred, cnn_loss = cnn_predict_function( label, image_name, img, cnn_model_path )
                st.write('CNN')
                st.write(f'Actual label: {label}')
                st.write(f'Predicted label: {cnn_pred.item()}')
                
            with col2:
                ### Model ...... Mobilenet
                mobnet_model_path = 'pre-trained-results/Mobilenet/_best.pth'
                mb_pred, mb_loss = mobnet_predict_function( label, image_name, img, mobnet_model_path )
                st.write('MobileNet')
                st.write(f'Actual label: {label}')
                st.write(f'Predicted label: {mb_pred.item()}')
                
            with col3:
                ### Model ...... ViT
                vit_model_path = 'pre-trained-results/ViT/_best.pth'
                vit_pred, vit_loss = vit_predict_function( label, image_name, img, vit_model_path )
                st.write('ViT')
                st.write(f'Actual label: {label}')
                st.write(f'Predicted label: {vit_pred.item()}')
            
    elif option2 == 'Display individual metrics':
        display_function( cnn_history_path, mob_history_path, vit_history_path, merged = False )
        
    elif option2 == 'Display combined metrics':
        display_function( cnn_history_path, mob_history_path, vit_history_path, merged = True )
        
        