# ImageClassifierUsingCNN
## Lion vs Elephant Image Classifier using CNN
This project is a Convolutional Neural Network (CNN)-based image classification system that distinguishes between images of lions and elephants. It is built using TensorFlow/Keras and trained on a custom dataset of lion and elephant images organized into separate folders.

### Project Overview
Model Type: Sequential CNN

Framework: TensorFlow / Keras

Input: RGB Images resized to 128x128 pixels

Output: Binary classification — predicts whether an image contains a lion or an elephant

### Dataset Structure
The dataset should be organized in the following directory format:

Data/

lion/

lion_1.jpg

lion_2.jpg

...

elephant/

elephant_1.jpg

elephant_2.jpg

...

Each class folder contains relevant images in .jpg, .png, or other supported formats. These folders are used by Keras’ ImageDataGenerator for automatic labeling and preprocessing.

### Model Architecture
The CNN model consists of:

3 Convolutional layers with ReLU activation

3 MaxPooling layers for downsampling

1 Flatten layer to convert feature maps to a vector

1 Dense layer with 256 units and ReLU activation

1 Output Dense layer with 1 neuron and sigmoid activation (for binary classification)

The model is compiled using:

Loss function: binary_crossentropy

Optimizer: adam

Metric: accuracy

### Running the Project
Clone the repository to your local system.

Install the required libraries:

tensorflow

numpy

opencv-python

matplotlib

Organize your dataset in the Data/ folder as described above.

Run the training script (or the code block containing model.fit(...)) to train the CNN model.

After training, the model is saved as lion_elephant_cnn_model.keras.

Testing with a New Image
You can test the model on any new image (not in the training dataset) using this process:

Load the model using load_model()

Load and preprocess the new image to 128x128 and normalize pixel values

Use model.predict() on the image

Use a threshold of 0.5 to classify:

Output < 0.5 → Elephant

Output >= 0.5 → Lion

A sample prediction script is provided in the repository.

### Model Performance
Achieved high accuracy (~95% or higher depending on image quality and dataset diversity)

Performs well on unseen images of lions and elephants

Output is probabilistic, allowing for confidence-based decisions

### Future Improvements
Add support for more animal classes (e.g., tiger, cheetah)

Improve robustness by training on larger and more diverse datasets

Deploy the model as a web app using Flask or Streamlit

Add support for real-time webcam-based classification



