import pandas as pd
import graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report

# Loading the dataset
df = pd.read_csv(r'C:\Fall 23\ENPM 808L\Week2 report\IrisNew.csv')
new_target = LabelEncoder()
df['target'] = new_target.fit_transform(df['Class'])
# Rename the columns in the 'inputs' DataFrame
inputs = df.drop(['Class', 'target'], axis='columns')
inputs.columns = ['Sepal Length(cm)', 'Sepal Width(cm)', 'Petal Length(cm)', 'Petal Width(cm)']
target = df['target']
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.33, random_state=42)
# Change the purity measure using the 'criterion' parameter
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("The score of the model on the test data is:", score)
# Evaluate decision tree model
y_tree_pred = model.predict(X_test)
tree_classification_report = classification_report(y_test, y_tree_pred)
print("\nClassification Report for Decision Tree Model based on criterion='entropy':")
print(tree_classification_report)

# Export the decision tree to a DOT file
dot_data = export_graphviz(model, out_file=None, feature_names=inputs.columns,
                           class_names=df['Class'],filled=True, rounded=True, 
                           special_characters=True)
# Creating a Graphviz object from the DOT data
graph = graphviz.Source(dot_data)
graph.view()# Save the decision tree visualization and display it

