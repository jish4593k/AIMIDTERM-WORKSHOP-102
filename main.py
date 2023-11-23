import pandas as pd
import numpy as np
import operator
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import tkinter as tk
from tkinter import messagebox

# Load the Iris dataset
columnHeaders = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
dataSetIris = pd.read_csv("iris.xlsx", names=columnHeaders)

# Euclidean Distance calculation function
def EuclideanDistance(node1, node2, length):
    distance = np.sum(np.square(node1 - node2))
    return np.sqrt(distance)

# K-nearest neighbors - calculate the nearest neighbors function
def KNN(trainingSet, testInstance, k):
    distances = {}
    length = len(testInstance)

    # Calculate Euclidean distance between testInstance and each training instance
    for x in range(len(trainingSet)):
        distance = EuclideanDistance(testInstance.values.flatten(), trainingSet.iloc[x, :-1].values, length)
        distances[x] = distance

    # Sort distances and get the indices of the first k neighbors
    sortedDistances = sorted(distances.items(), key=operator.itemgetter(1))
    neighbors = [x[0] for x in sortedDistances[:k]]

    # Count the occurrences of each class in the neighbors
    counts = {"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0}
    for x in neighbors:
        response = trainingSet.iloc[x][-1]
        counts[response] += 1

    # Sort the counts in descending order
    sortedVotes = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0], neighbors

# Linear Regression for petal length prediction
def linear_regression(data):
    X = data[['sepal_length', 'sepal_width', 'petal_width']]
    y = data['petal_length']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction for a new test instance
    test_instance = pd.DataFrame([[5.1, 3.5, 1.4]], columns=['sepal_length', 'sepal_width', 'petal_width'])
    prediction = model.predict(test_instance)
    return prediction[0]

# GUI using Tkinter
def run_knn():
    test_instance = pd.DataFrame([[float(entry_sepal_length.get()), float(entry_sepal_width.get()),
                                   float(entry_petal_length.get()), float(entry_petal_width.get())]])

    k_value = int(entry_k.get())
    result, neighbors = KNN(dataSetIris, test_instance, k_value)
    messagebox.showinfo("KNN Result", f"Flower type: {result}\nNeighbors: {neighbors}")

def run_linear_regression():
    prediction = linear_regression(dataSetIris)
    messagebox.showinfo("Linear Regression Result", f"Predicted Petal Length: {prediction}")

# Visualize the dataset using Plotly
fig = px.scatter_3d(dataSetIris, x='sepal_length', y='sepal_width', z='petal_width', color='type')
fig.show()

# Tkinter GUI
root = tk.Tk()
root.title("Iris Flower Classifier")

label_sepal_length = tk.Label(root, text="Sepal Length:")
label_sepal_width = tk.Label(root, text="Sepal Width:")
label_petal_length = tk.Label(root, text="Petal Length:")
label_petal_width = tk.Label(root, text="Petal Width:")
label_k = tk.Label(root, text="K Value:")

entry_sepal_length = tk.Entry(root)
entry_sepal_width = tk.Entry(root)
entry_petal_length = tk.Entry(root)
entry_petal_width = tk.Entry(root)
entry_k = tk.Entry(root)

button_knn = tk.Button(root, text="Run KNN", command=run_knn)
button_linear_regression = tk.Button(root, text="Run Linear Regression", command=run_linear_regression)

label_sepal_length.grid(row=0, column=0)
label_sepal_width.grid(row=1, column=0)
label_petal_length.grid(row=2, column=0)
label_petal_width.grid(row=3, column=0)
label_k.grid(row=4, column=0)

entry_sepal_length.grid(row=0, column=1)
entry_sepal_width.grid(row=1, column=1)
entry_petal_length.grid(row=2, column=1)
entry_petal_width.grid(row=3, column=1)
entry_k.grid(row=4, column=1)

button_knn.grid(row=5, column=0, columnspan=2, pady=10)
button_linear_regression.grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()
