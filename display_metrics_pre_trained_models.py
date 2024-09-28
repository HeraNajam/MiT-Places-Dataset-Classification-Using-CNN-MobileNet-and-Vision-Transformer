
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def plot_function( cnn_data, model_name, metric_name ):
    
    plt.figure(figsize=(6, 4))
    
    plt.plot(cnn_data[ 'train_' + metric_name ], label = 'train', marker='o')
    plt.plot(cnn_data[ 'valid_' + metric_name ], label = 'valid',  marker='*')
    
    plt.title(f'{model_name} - {metric_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    
    plt.legend()
    plt.grid()

    st.pyplot(plt)
    
def plot_function_combined( cnn_data, mob_data, vit_data, metric_name, desc ):
    
    
    plt.figure(figsize=(6, 4))
    
    plt.plot(cnn_data[ desc + '_' + metric_name ], label = 'CNN', marker='o')
    plt.plot(mob_data[ desc + '_' + metric_name ], label = 'Mobilenet',  marker='*')
    plt.plot(vit_data[ desc + '_' + metric_name ], label = 'ViT',  marker='*')
    
    plt.title(f'{desc} - {metric_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    
    plt.legend()
    plt.grid()

    st.pyplot(plt)

def load_json( path ):
    
    # Open JSON file
    f = open(path)
    # returns JSON object as a dictionary
    data = json.load(f)
    
    return data    
    
def display_function( cnn_hist_path, mob_hist_path, vit_hist_path, merged ):
    
    # Model.... CNN
    cnn_data = load_json( cnn_hist_path )
    
    # Model.... Mobilenet
    mob_data = load_json( mob_hist_path )
    
    # Model.... ViT
    vit_data = load_json( vit_hist_path )
    
    if merged == False:
        
        st.header('CNN Metrics')
        col1, col2 = st.columns(2)
        with col1:
            plot_function( cnn_data, 'CNN', 'loss' )
        with col2:
            plot_function( cnn_data, 'CNN', 'accuracy' )
            
        st.header('MobileNet Metrics')
        col1, col2 = st.columns(2)
        with col1:
            plot_function( mob_data, 'Mobilenet', 'loss' )
        with col2:
            plot_function( mob_data, 'Mobilenet', 'accuracy' )
            
        st.header('ViT Metrics')
        col1, col2 = st.columns(2)
        with col1:
            plot_function( vit_data, 'ViT', 'loss' )
        with col2:
            plot_function( vit_data, 'ViT', 'accuracy' )
    
    elif merged == True:
        
        #### Train results combined
        st.header('Training Metrics')
        
        col1, col2 = st.columns(2)
        with col1:
            plot_function_combined( cnn_data, mob_data, vit_data,  'loss', desc = 'train' )
        with col2:
            plot_function_combined( cnn_data, mob_data, vit_data,  'accuracy', desc = 'train' )
        
        #### Validation results combined
        st.header('Validation Metrics')
        
        col1, col2 = st.columns(2)
        with col1:
            plot_function_combined( cnn_data, mob_data, vit_data, 'loss', desc = 'valid' )
        with col2:
            plot_function_combined( cnn_data, mob_data, vit_data, 'accuracy', desc = 'valid' )
    

                
                

