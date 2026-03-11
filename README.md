# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The objective of this project is to create a CNN that can categorize images of fashion items from the Fashion MNIST dataset. This dataset includes grayscale images of clothing and accessories such as T-shirts, trousers, dresses, and footwear. The task is to accurately predict the correct category for each image while ensuring the model is efficient and robust.

1.Training data: 60,000 images

2.Test data: 10,000 images

3.Classes: 10 fashion categories

The CNN consists of multiple convolutional layers with activation functions, followed by pooling layers, and ends with fully connected layers to output predictions for all 10 categories.

## Neural Network Model

<img width="962" height="468" alt="image" src="https://github.com/user-attachments/assets/e81eb9fa-27d9-4ba2-8ce6-73bdd2fc4233" />

## DESIGN STEPS

### STEP 1:
Import the necessary libraries such as NumPy, Matplotlib, and PyTorch.

### STEP 2:
Load and preprocess the dataset:

Resize images to a fixed size (128×128).
Normalize pixel values to a range between 0 and 1.
Convert labels into numerical format if necessary.
### STEP 3:
Define the CNN Architecture, which includes:

Input Layer: Shape (8,128,128)
Convolutional Layer 1: 8 filters, kernel size (16×16), ReLU activation
Max-Pooling Layer 1: Pool size (2×2)
Convolutional Layer 2: 24 filters, kernel size (8×8), ReLU activation
Max-Pooling Layer 2: Pool size (2×2)
Fully Connected (Dense) Layer:
First Dense Layer with 256 neurons
Second Dense Layer with 128 neurons
Output Layer for classification
### STEP 4:
Define the loss function (e.g., Cross-Entropy Loss for classification) and optimizer (e.g., Adam or SGD).

### STEP 5:
Train the model by passing training data through the network, calculating the loss, and updating the weights using backpropagation.

### STEP 6:
Evaluate the trained model on the test dataset using accuracy, confusion matrix, and other performance metrics.

### STEP 7:
Make predictions on new images and analyze the results.



## PROGRAM

### Name:JISHA BOSSNE SJ
### Register Number:212224230106
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # Convolution Layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Convolution Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 28x28 → 14x14
        x = self.pool(self.relu(self.conv2(x)))   # 14x14 → 7x7
        x = x.view(-1, 64 * 7 * 7)                # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        
        print('Name: JISHA BOSSNE SJ       ')
        print('Register Number:  212224230106     ')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

<img width="496" height="216" alt="image" src="https://github.com/user-attachments/assets/4862d760-7047-4380-896d-ead7becdb5e4" />

### Confusion Matrix

<img width="907" height="795" alt="image" src="https://github.com/user-attachments/assets/2a4afebf-a86b-4f9e-8ebd-2b29400b6d0e" />

### Classification Report

<img width="643" height="413" alt="image" src="https://github.com/user-attachments/assets/e73fff89-9176-48c4-87d1-3114ed2dc844" />

### New Sample Data Prediction

<img width="576" height="562" alt="image" src="https://github.com/user-attachments/assets/9955f509-1731-4b48-af0b-e35892295c5d" />

## RESULT
The Convolutional Neural Network (CNN) was successfully implemented for image classification. The model was trained on the dataset, and its performance was evaluated using accuracy metrics, confusion matrix, and classification report. Predictions on new sample images were verified, confirming the model's effectiveness in classifying images.
