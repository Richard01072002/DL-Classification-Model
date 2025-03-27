# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model
![image](https://github.com/user-attachments/assets/700f8c8f-3c12-490b-9c5e-2d6102323145)

## DESIGN STEPS
### STEP 1: 
Load the Iris dataset, split it into train-test sets, standardize features, and convert them to tensors.

### STEP 2: 
Use TensorDataset and DataLoader for efficient data handling.

### STEP 3: 
Define IrisClassifier with input, hidden (ReLU), and output layers.

### STEP 4: 
Train using CrossEntropyLoss and Adam optimizer for 100 epochs.

### STEP 5: 
Evaluate with accuracy, confusion matrix, and classification report, and visualize results.

### STEP 6: 
Predict a sample input and compare with the actual class.

## PROGRAM

### Name: RICHARDSON A

### Register Number: 212222233005

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

iris=load_iris()
x=iris.data
y=iris.target

df=pd.DataFrame(x,columns=iris.feature_names)
df['target']=y

print("\nFirst 5 rows of the dataset:\n",df.head())
print("\nLast 5 rows of the dataset:\n",df.tail())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

x_train=torch.tensor(x_train,dtype=torch.float32)
x_test=torch.tensor(x_test,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.long)
y_test=torch.tensor(y_test,dtype=torch.long)

train_dataset=TensorDataset(x_train, y_train)
test_dataset=TensorDataset(x_test, y_test)
train_loader=DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=16)


class IrisClassifier(nn.Module):
  def __init__(self,input_size):
    super(IrisClassifier,self).__init__()
    self.fc1=nn.Linear(input_size, 16)
    self.fc2=nn.Linear(16, 8)
    self.fc3=nn.Linear(8, 3)

  def forward(self, x):
    x=F.relu(self.fc1(x))
    x=F.relu(self.fc2(x))
    return self.fc3(x)

def train_model(model, train_loader, criterion, optimizer, epochs):
  for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(x_batch)
      loss=criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()
    if (epoch + 1) % 10 == 0:
      print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

model = IrisClassifier(input_size=x_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, epochs=100)

model.eval()
predictions, actuals = [], []
with torch.no_grad():
  for x_batch, y_batch in test_loader:
    outputs = model(x_batch)
    _, predicted = torch.max(outputs, 1)
    predictions.extend(predicted.numpy())
    actuals.extend(y_batch.numpy())

accuracy=accuracy_score(actuals, predictions)
confusion_matrix=confusion_matrix(actuals, predictions)
classification_report=classification_report(actuals, predictions)


print("Name: SHYAM S")
print("Register Number: 212223240156")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", confusion_matrix)
print("Classification Report:\n", classification_report)

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix, annot=True,cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

sample_input = x_test[5].unsqueeze(0)
with torch.no_grad():
  output = model(sample_input)
  predicted_class_index = torch.argmax(output[0]).item()
  predicted_class_label = iris.target_names[predicted_class_index]

print("\nName: SHYAM S")
print("\nRegister Name: 212223240156")
print(f"Predicted class for sample input: {predicted_class_label}")
print(f"Actual class for sample input: {iris.target_names[y_test[5].item()]}")
```

### Dataset Information
![image](https://github.com/user-attachments/assets/04cc57e7-4822-462f-ae24-21ae9c5de43c)

### OUTPUT

## Confusion Matrix
![image](https://github.com/user-attachments/assets/e5413946-2035-4c7e-99df-16b0b35d2ec0)

![image](https://github.com/user-attachments/assets/81eabe9c-2a4c-46ca-bb3f-511301e3cef0)

## Classification Report
![image](https://github.com/user-attachments/assets/cb1a385c-a190-4c0f-9c6a-97a898bcebad)

### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/6bc81661-540c-4505-b330-a0b152b21303)

## RESULT

Thus, a neural network classification model has been successfully built.
