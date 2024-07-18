# Plant-Disease-Prediction-CNN-
This project aims to develop a Convolutional Neural Network (CNN) to accurately predict plant diseases from images. The model is trained on a dataset of plant images with labeled diseases and uses advanced data augmentation techniques and a tuned model architecture to enhance performance.
ey Components
Data Curation:

Collection and organization of the dataset containing images of healthy and diseased plants.
Ensuring the dataset is properly formatted and split into training, validation, and test sets.
Data Exploration and Visualization:

Use Seaborn for insightful visualizations to understand the distribution of image sizes and sample images from different classes.
Example visualizations include joint plots of image dimensions and sample image grids.
Data Augmentation:

Enhanced data augmentation techniques to artificially increase the size and variability of the dataset.
Techniques include rotation, width and height shifts, shear transformations, zoom, and horizontal flips.
Model Architecture:

Construction of a CNN model with multiple convolutional and pooling layers to extract features from the images.
Addition of dropout layers to prevent overfitting and dense layers for classification.
Model compiled with Adam optimizer and binary cross-entropy loss function.
Training and Evaluation:

Training the model on augmented data with a specified number of epochs and batch size.
Evaluation of the model's performance using validation data.
Visualization of training and validation accuracy/loss over epochs using Seaborn.
Steps to Execute the Project
Import Dependencies:

Import necessary libraries including TensorFlow, Keras, NumPy, Matplotlib, and Seaborn.
Set Up Data Generators:

Use ImageDataGenerator from Keras for data augmentation and to create training and validation generators.
Define and Compile the Model:

Define a sequential model with convolutional, pooling, dropout, and dense layers.
Compile the model with the Adam optimizer and binary cross-entropy loss.
Train the Model:

Fit the model using the training data generator and validate using the validation data generator.
Monitor performance and adjust parameters as necessary.
Evaluate and Visualize Results:

Evaluate the final model performance on the test set.
Use Seaborn to visualize the training history, including accuracy and loss curves.
Conclusion
This project provides a comprehensive approach to building a robust CNN-based image classifier for plant disease prediction. By utilizing advanced data augmentation techniques and a well-tuned model architecture, the project aims to achieve high accuracy and reliability in identifying plant diseases from images.
