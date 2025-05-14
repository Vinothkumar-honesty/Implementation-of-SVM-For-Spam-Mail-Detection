# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the spam dataset and handle encoding properly.
2. Display basic information and check for null values.
3. Extract the message text as features (x) and labels (y) for classification.
4. Split the dataset into training and testing sets.
5. Convert the text data into numerical vectors using CountVectorizer.
6. Train an SVM classifier on the transformed training data.
7. Predict on test data and evaluate model accuracy using accuracy_score.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VINOTHKUMAR R
RegisterNumber:  212224040361
*/
```
```
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
data = pd.read_csv("/content/spam.csv", encoding="Windows-1252")
data.head()
```
```
data.info()
```
```
data.isnull().sum()
```
```
# separating the features and labels
x = data["v2"].values  # text messages
y = data["v1"].values  # labels: spam or ham
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```
```
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
```
```
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
y_pred
```
```
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}%")
```



## Output:
**Head Values**

![1](https://github.com/user-attachments/assets/fe3d5235-db0e-40b8-8c69-f15e89d4f166)


**Dataframe Info**

![2](https://github.com/user-attachments/assets/2319c44f-4825-4961-8d1f-983bab13a833)


**Sum - Null Values**


![3](https://github.com/user-attachments/assets/4939e0fb-f364-495f-a74c-084ee0fb0849)


**Training the model**

![Screenshot 2025-05-14 093215](https://github.com/user-attachments/assets/aa6e8b63-3df0-46f5-8853-c33569ead761)


**Predicting the test data**

![4](https://github.com/user-attachments/assets/ff3fbd2e-2327-4cde-8c1b-2a7ab1800a4a)



**Accuracy**

![5](https://github.com/user-attachments/assets/33e1ddea-740c-441b-b3cd-cf2a86a982d6)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
