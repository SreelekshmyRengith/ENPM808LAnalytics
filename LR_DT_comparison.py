import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load the Iris dataset
iris = pd.read_csv(r'C:\Fall 23\ENPM 808L\Week2 report\IrisNew.csv')
iris_target=LabelEncoder()
iris['Target'] = iris_target.fit_transform(iris['Class'])
inputs=iris.drop(['Class', 'Target'], axis='columns')
target = iris['Target']
inputs.columns = ['Sepal Length(cm)', 'Sepal Width(cm)', 'Petal Length(cm)', 'Petal Width(cm)']

# Create logistic regression model
logr_model = LogisticRegression(solver='lbfgs', multi_class='auto')
# Create decision tree classifier model
tree_model = DecisionTreeClassifier()

# Train-test split for Iris data
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3, random_state=42)

# Train the logistic regression model on Iris data
logr_model.fit(X_train, y_train)

# Train the decision tree model on Iris data
tree_model.fit(X_train, y_train)

# Evaluate logistic regression model
y_logr_pred = logr_model.predict(X_test)
logr_classification_report = classification_report(y_test, y_logr_pred)

# Evaluate decision tree model
y_tree_pred = tree_model.predict(X_test)
tree_classification_report = classification_report(y_test, y_tree_pred)

# Compare the results
print("Classification Report for Logistic Regression Model:")
print(logr_classification_report)

print("\nClassification Report for Decision Tree Model:")
print(tree_classification_report)

# Create a ranking of the results of the logistic regression model using predict_proba method
logr_probabilities = logr_model.predict_proba(X_test)
logr_ranking = pd.DataFrame(logr_probabilities,columns=iris_target.classes_)

print("\nRanking of Results for Logistic Regression Model:")
print(logr_ranking.head())

# Rank two new data records using the logistic regression model
new_data = [[5.8, 2.8, 5.1, 2.4], [6.0, 2.2, 4.0, 1.0]]
new_data_predictions = logr_model.predict(new_data)
new_data_probabilities = logr_model.predict_proba(new_data)

print("\nPredictions for New Data:")
for i, data_point in enumerate(new_data):
    print(f"Data Point {i + 1}: {data_point} => Predicted Class: {iris_target.classes_[new_data_predictions[i]]}")
    print(f"Class Probabilities: {dict(zip(iris_target.classes_, new_data_probabilities[i]))}\n")
