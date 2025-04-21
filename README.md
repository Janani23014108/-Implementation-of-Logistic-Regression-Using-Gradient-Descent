# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary.
6. Define a function to predict the Regression value.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: J.JANANI
RegisterNumber: 212223230085 
*/
```

```
import pandas as pd
import numpy as np
dataset=pd.read_csv("Placement_Data.csv")
dataset
```
![429084908-b9263b8e-85bc-4215-a992-3dcac4bcc4df](https://github.com/user-attachments/assets/fcfab480-b3ca-45c5-9bf1-3d65ba577631)

```
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```

![429084986-10e93552-634f-4b90-9f1c-93121264344d](https://github.com/user-attachments/assets/d9965e24-5e0b-4053-aa0e-5301f8a56a10)

```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
![429085058-94a85212-4dae-48b0-868a-3c7f13a38c16](https://github.com/user-attachments/assets/cc263ecf-94ff-4bd4-aa65-512a633a185b)

```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
```
![429085194-dc9043b2-048c-4b79-afbd-f180ae646d2f](https://github.com/user-attachments/assets/db3b870e-8f62-4b46-9f83-ecb0bc0f357e)

```
theta = np.random.randn(X.shape[1])
y =Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h= sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations = 1000)
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred

y_pred = predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)
```

![429085280-b434ba72-7e3b-4209-bc0b-c7448c3632b2](https://github.com/user-attachments/assets/cabb0090-a308-4bed-8622-ef8fef9d13fd)


```
print(y_pred)
```
![429085360-35d769b2-75a3-43b6-bd19-3d295b2305ab](https://github.com/user-attachments/assets/3528b460-d655-434c-b14e-54e46dd9b6f5)

```
print(y)
```
![429085505-c5652fd9-2c2f-4cca-9549-c2cb73dbb0b8](https://github.com/user-attachments/assets/158953e8-1bd2-40fd-a4cc-57d42bbcf918)


```
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```

![429085529-b599f115-5fe2-440b-ad06-527511a02148](https://github.com/user-attachments/assets/4508311c-ea58-4c4b-b64e-71abed8e11cb)




```
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```

![429085596-099f4f97-fe2b-4257-b347-b3e9e848e0e0](https://github.com/user-attachments/assets/03b311b4-2506-4893-a341-433964dca8ed)


## Result:

Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

