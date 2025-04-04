# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
#### **STEP 1: Data Collection and Preprocessing**  
- Load the **Iris dataset** from `iris.csv`.  
- Check for **missing values** and **duplicates** (if necessary).  
- Split data into **features (X)** and **labels (y)**.  
- Convert categorical labels into numerical form if needed.  

#### **STEP 2: Data Visualization**  
- Use **scatter plots** to visualize relationships between features.  
- Assign different colors for different Iris species.  
- Identify any outliers or patterns in the data.  

#### **STEP 3: Train-Test Split and Data Preparation**  
- Split the dataset into **training (80%)** and **testing (20%)** sets.  
- Convert data into **PyTorch tensors** for model training.  
- Create a **custom PyTorch Dataset** for efficient data loading.  

#### **STEP 4: Model Definition and Compilation**  
- Define a **Neural Network model** using `torch.nn.Module`.  
- Include:
  - **Input layer** (4 neurons for 4 features)  
  - **Two hidden layers** with activation functions  
  - **Output layer** (3 neurons for 3 classes)  
- Choose an **appropriate loss function** (`CrossEntropyLoss`).  
- Select an **optimizer** (`Adam` with a learning rate of 0.01).  

#### **STEP 5: Model Training**  
- Train the model for **100 epochs**.  
- Track **loss reduction** over epochs.  
- Update weights using **backpropagation and optimization**.  

#### **STEP 6: Model Evaluation & Prediction**  
- Evaluate model performance on the **test dataset**.  
- Calculate **accuracy** and **final loss**.  
- Save the trained model using `torch.save()`.  
- Load the model and **predict on a new (mystery) Iris flower**.  
- Visualize the **mystery iris** on the scatter plot.


## PROGRAM

### Name: RICHARDSON A

### Register Number: 212222233005

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)    # input layer
        self.fc2 = nn.Linear(h1, h2)            # hidden layer
        self.out = nn.Linear(h2, out_features)  # output layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
# Instantiate the Model class using parameter defaults:
torch.manual_seed(32)
model = Model()

df = pd.read_csv('iris.csv')
df.head()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
fig.tight_layout()

plots = [(0,1),(2,3),(0,2),(1,3)]
colors = ['b', 'r', 'g']
labels = ['Iris setosa','Iris virginica','Iris versicolor']

for i, ax in enumerate(axes.flat):
    for j in range(3):
        x = df.columns[plots[i][0]]
        y = df.columns[plots[i][1]]
        ax.scatter(df[df['target']==j][x], df[df['target']==j][y], color=colors[j])
        ax.set(xlabel=x, ylabel=y)

fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
plt.show()

X = df.drop('target',axis=1).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=33)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
# y_train = F.one_hot(torch.LongTensor(y_train))  # not needed with Cross Entropy Loss
# y_test = F.one_hot(torch.LongTensor(y_test))
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

trainloader = DataLoader(X_train, batch_size=60, shuffle=True)

testloader = DataLoader(X_test, batch_size=60, shuffle=False)

# FOR REDO
torch.manual_seed(4)
model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses.append(loss)
losses.append(loss.item)

epochs = 100
losses = []

for i in range(epochs):
    i+=1
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())
    
    # a neat trick to save screen space:
    if i%10 == 1:
        print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch');

# TO EVALUATE THE ENTIRE TEST SET
with torch.no_grad():
    y_val = model.forward(X_test)
    loss = criterion(y_val, y_test)
print(f'{loss:.8f}')

correct = 0
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_val = model.forward(data)
        print(f'{i+1:2}. {str(y_val):38}  {y_test[i]}')
        if y_val.argmax().item() == y_test[i]:
            correct += 1
print(f'\n{correct} out of {len(y_test)} = {100*correct/len(y_test):.2f}% correct')


torch.save(model.state_dict(), 'IrisDatasetModel.pt')


new_model = Model()
new_model.load_state_dict(torch.load('IrisDatasetModel.pt'))
new_model.eval()


with torch.no_grad():
    y_val = new_model.forward(X_test)
    loss = criterion(y_val, y_test)
print(f'{loss:.8f}')


mystery_iris = torch.tensor([5.6,3.7,2.2,0.5])


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
fig.tight_layout()

plots = [(0,1),(2,3),(0,2),(1,3)]
colors = ['b', 'r', 'g']
labels = ['Iris setosa','Iris virginica','Iris versicolor','Mystery iris']

for i, ax in enumerate(axes.flat):
    for j in range(3):
        x = df.columns[plots[i][0]]
        y = df.columns[plots[i][1]]
        ax.scatter(df[df['target']==j][x], df[df['target']==j][y], color=colors[j])
        ax.set(xlabel=x, ylabel=y)
        
    # Add a plot for our mystery iris:
    ax.scatter(mystery_iris[plots[i][0]],mystery_iris[plots[i][1]], color='y')
    
fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
plt.show()



with torch.no_grad():
    print(new_model(mystery_iris))
    print()
    print(labels[new_model(mystery_iris).argmax()])
```

### Dataset Information
<img width="818" alt="Screenshot 2025-03-29 at 11 14 51 AM" src="https://github.com/user-attachments/assets/a72bf958-0c96-43de-b1c3-8b378a209326" />

### OUTPUT



<img width="287" alt="Screenshot 2025-03-29 at 11 16 32 AM" src="https://github.com/user-attachments/assets/ac6393e5-6918-493c-84a7-b13b8f32072b" />

<img width="657" alt="Screenshot 2025-03-29 at 11 16 51 AM" src="https://github.com/user-attachments/assets/23cd6474-cbf5-4321-b5ee-2b0c8a4e9914" />

<img width="409" alt="Screenshot 2025-03-29 at 11 19 41 AM" src="https://github.com/user-attachments/assets/711aebea-ef69-485e-8a42-44a6ddf23ff6" />

## Confusion Matrix

Include confusion matrix here

## Classification Report
Include classification report here

### New Sample Data Prediction
Include your sample input and output here

## RESULT
This project demonstrates how a simple deep learning model can accurately classify iris flowers based on four numerical features. The same approach can be expanded for more complex classification problems in domains like healthcare, agriculture, and image recognition.
