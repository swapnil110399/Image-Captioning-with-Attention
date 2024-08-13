# Image Captioning with Attention

## Overview
This project implements an image captioning system that uses a combination of convolutional neural networks (CNN) and gated recurrent units (GRU) with multi-head attention to generate descriptive captions for images. This approach enhances the standard CNN-LSTM model by incorporating more robust attention mechanisms and recurrent units for improved context capturing in captions.

## Features
- **Encoder-Decoder Architecture**: Utilizes ResNet-50 as the backbone for the encoder and a GRU-based decoder.
- **Multi-Head Attention**: Implements a custom multi-head attention mechanism to focus on different parts of the image.
- **Dataset**: Trained on the Flickr8k dataset, which consists of 8,000 images each paired with five different captions.

## Requirements
To run this project, you need the following libraries:
- Python 3.8+
- PyTorch 1.7+
- torchvision
- nltk
- PIL
- matplotlib

## Dataset
The Flickr8k dataset is used for training the image captioning model. This dataset consists of 8,000 images each paired with five different captions, which is ideal for training and testing our model.

### Download the Dataset
You can download the images and annotations from the following links:
- **Images**: [Download Flickr8k_Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
- **Annotations**: [Download Flickr8k_Text](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)

