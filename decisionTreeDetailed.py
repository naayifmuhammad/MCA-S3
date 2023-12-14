from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
x = iris.data
y = iris.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Create a Decision Tree classifier with Gini index
clf_gini = DecisionTreeClassifier()
clf_gini = clf_gini.fit(x_train, y_train)
y_pred_gini = clf_gini.predict(x_test)

# Calculate and print accuracy using Gini index
accuracy_gini = metrics.accuracy_score(y_test, y_pred_gini)
print("Accuracy using Gini index: ", accuracy_gini)

# Print accuracy on train data using Gini index
train_accuracy_gini = metrics.accuracy_score(y_train, clf_gini.predict(x_train))
print('Accuracy on train data using Gini index:', train_accuracy_gini)

# Create a Decision Tree classifier with entropy
clf_entropy = DecisionTreeClassifier(criterion='entropy')
clf_entropy = clf_entropy.fit(x_train, y_train)
y_pred_entropy = clf_entropy.predict(x_test)

# Calculate and print accuracy using entropy
accuracy_entropy = metrics.accuracy_score(y_test, y_pred_entropy)
print("Accuracy using entropy: ", accuracy_entropy)

# Print accuracy on train data using entropy
train_accuracy_entropy = metrics.accuracy_score(y_train, clf_entropy.predict(x_train))
print('Accuracy on train data using entropy:', train_accuracy_entropy)

# Plot the decision tree with Gini index
plt.figure(figsize=(15, 15))
plot_tree(clf_gini, fontsize=10, filled=True, rounded=True, class_names=iris.target_names, feature_names=iris.feature_names)
plt.title('Decision Tree with Gini Index')
plt.show()

# Plot the decision tree with entropy
plt.figure(figsize=(15, 15))
plot_tree(clf_entropy, fontsize=10, filled=True, rounded=True, class_names=iris.target_names, feature_names=iris.feature_names)
plt.title('Decision Tree with Entropy')
plt.show()
