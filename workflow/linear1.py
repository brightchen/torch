import torch
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(1)
y = weight * X + bias

print("======= data: ========")
print(X[:4])
print(y[:4])

#split the dataset into train and validation
ratio = 0.7
train_end = int(ratio * len(X))
X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[train_end:], y[train_end:]

print("======= split data into training and test dataset: ========")
print(f"train_X size: {len(X_train)}; test_X size: {len(X_test)}")
print(f"train_y size: {len(y_train)}; test_y size: {len(y_test)}")

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14});

  # need this to show the graph when running in console
  plt.show()

plot_predictions();