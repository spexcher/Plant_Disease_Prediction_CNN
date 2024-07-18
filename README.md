# Plant Disease Prediction Using CNN Image Classifier

## Project Overview

This project aims to develop a Convolutional Neural Network (CNN) to accurately predict plant diseases from images. The model is trained on a dataset of plant images with labeled diseases and uses advanced data augmentation techniques and a tuned model architecture to enhance performance.

## Key Components

### 1. Data Curation

- Collection and organization of the dataset containing images of healthy and diseased plants.
- Ensuring the dataset is properly formatted and split into training, validation, and test sets.

### 2. Data Exploration and Visualization

- Use Seaborn for insightful visualizations to understand the distribution of image sizes and sample images from different classes.
- Example visualizations include joint plots of image dimensions and sample image grids.

### 3. Data Augmentation

- Enhanced data augmentation techniques to artificially increase the size and variability of the dataset.
- Techniques include rotation, width and height shifts, shear transformations, zoom, and horizontal flips.

### 4. Model Architecture

- Construction of a CNN model with multiple convolutional and pooling layers to extract features from the images.
- Addition of dropout layers to prevent overfitting and dense layers for classification.
- Model compiled with Adam optimizer and binary cross-entropy loss function.

### 5. Training and Evaluation

- Training the model on augmented data with a specified number of epochs and batch size.
- Evaluation of the model's performance using validation data.
- Visualization of training and validation accuracy/loss over epochs using Seaborn.

## Steps to Execute the Project

1. **Setup Kaggle Account and Obtain API Key**:
   - Create a Kaggle account and download the `kaggle.json` API key.
   - Configure the path to the key in your notebook's `kaggle_secrets`.

2. **Import Dependencies**:
   - Import necessary libraries: TensorFlow, Keras, NumPy, Matplotlib, and Seaborn.

3. **Set Up Data Generators**:
   - Utilize `ImageDataGenerator` from Keras for data augmentation.
   - Create training and validation generators using prepared datasets.

4. **Define and Compile the Model**:
   - Build a sequential model with convolutional, pooling, dropout, and dense layers.
   - Compile the model using the Adam optimizer and binary cross-entropy loss.

5. **Train the Model**:
   - Fit the model with the training data generator.
   - Validate using the validation data generator.
   - Monitor performance and adjust parameters based on training metrics.

6. **Evaluate and Visualize Results**:
   - Evaluate the final model performance on a separate test set.
   - Visualize training history with Seaborn to assess accuracy and loss curves.
  
7. **That's it. Thank you for viewing my Project..spexcher**

## Conclusion

This project aims to develop a robust CNN-based image classifier for predicting plant diseases. By employing advanced data augmentation techniques and optimizing model architecture, the goal is to achieve high accuracy in identifying plant diseases from images.
