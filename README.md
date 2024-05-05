Title: Plant Disease Classification using Deep Learning

Description: This project aims to classify plant diseases using deep learning techniques. It utilizes convolutional neural networks (CNNs) to automatically identify the type of disease affecting plants based on images of plant leaves.

Problem setup:  Developing a deep learning model capable of accurately classifying plant diseases from images. This involves training a model using a dataset containing images of plants affected by various diseases, with the aim of creating a classifier that can identify the type of disease present in a given image.

Installation:
Clone the repository:
git clone 
cd plant-disease-classification
Install dependencies:
pip install -r requirements.txt

Dataset used: The dataset used in this project is the PlantVillage Dataset available on Kaggle. It contains images of various crop diseases and healthy plant samples. The dataset consists of:
 Segmented images (14 GB)
Color images (1.3 GB)
Grayscale images (1.3 GB)
To download the dataset, you can use the Kaggle API:
kaggle datasets download -d abdallahalidev/plantvillage-dataset

Usage:
Set up Kaggle API key by creating a kaggle.json file with your Kaggle username and API key.
Run the provided code to download and unzip the dataset.
Install the required packages.
Adjust the paths and parameters in the code as needed.
Run train.py to train the model.
Run predict.py to predict the class of a new image.
Evaluate the model's performance and visualize training/validation accuracy and loss.

Model Architecture: 
The model architecture consists of:
2 Conv2D layers
2 MaxPooling2D layers
1 Flatten layer
2 Dense layers
The model is compiled with the Adam optimizer and categorical cross-entropy loss.

Techniques:
Convolutional Neural Networks (CNNs): Deep learning models designed for processing grid-like data such as images, consisting of convolutional and pooling layers.
Data Augmentation: Technique involving applying transformations to training data, like rotation or flipping, to increase diversity and prevent overfitting.
Transfer Learning: Utilizing pre-trained neural network models and fine-tuning them on new tasks, leveraging knowledge from previous training.
Image Preprocessing: Preparing input images for training by resizing, normalizing, and potentially enhancing them to ensure suitability for model input.
Validation Split: Dividing the dataset into training and validation sets to monitor model performance during training and prevent overfitting.
Softmax Activation: Output layer activation function converting raw scores to probabilities for multi-class classification tasks.
Adam Optimizer: Optimization algorithm adapting learning rates for each parameter during training, known for efficiency and effectiveness.
Loss Function: Quantifies the difference between predicted and actual distributions of class labels, serving as the optimization objective during training.

Pre-existing components:
The pre-existing components are:
Data downloading and preprocessing: The code downloads a dataset from Kaggle using the Kaggle API and unzips it. This process likely involves using existing libraries and code snippets for handling datasets and file operations.
Sequential Model Definition: The code utilizes the Sequential model from TensorFlow's Keras API to define the neural network architecture. This is a common approach in deep learning projects and is readily available from TensorFlow.
ImageDataGenerator: The code uses ImageDataGenerator from Keras for data augmentation and generating batches of image data during training. This is a common technique in deep learning for efficiently feeding data to the model.
Model Compilation: The model is compiled using standard techniques in deep learning, including specifying the optimizer, loss function, and metrics, as well as calling the fit function on the model with training data.

New components: 
These are the new components that are added:
Data directory setup: Organizing the dataset into specific directories (e.g., train, validation) may have been customized for this particular project.
Prediction Function: A custom function predict_image_class is defined to predict the class of an image based on the trained model. This function takes an image path, preprocesses the image, and returns the predicted class name.
Class Indices Mapping: A mapping from class indices to class names is created and saved as a JSON file. This mapping is used in the prediction function to interpret the predicted class index.
Ground Truth Labels: The ground truth labels (y_true) are obtained from the classe attribute of the validation generator. This attribute contains the true class labels corresponding to the images in the validation set.
Precision, Recall, and F1 Score Calculation: These metrics are computed using the precision_recall_fscore_support function from Scikit-learn. This function computes precision, recall, and F1 score for each class individually and then averages them using the specified averaging strategy ('weighted' in this case).
Accuracy Calculation: The accuracy is computed using the accuracy_score function from Scikit-learn. It calculates the proportion of correctly predicted labels to the total number of samples in the validation set.
Accuracy: Accuracy is a measure of the overall correctness of the model's predictions. It represents the percentage of correctly classified samples out of all samples. It is also newly calculated.
Overall, these added components enable the evaluation of the model's performance using key classification metrics such as precision, recall, accuracy, and F1 score, providing insights into how well the model is performing across different classes and overall.

Results:
After training the model for 1 epochs, the validation accuracy reached approximately 82.11%. Visualizations of training and validation accuracy/loss are provided to assess model performance.

Acknowledgments:
PlantVillage Dataset
TensorFlow
Keras
Numpy
Pillow
Matplotlib
