import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

# Define the logistic_function after defining X_wiki
def logistic_function(x):
    return (1 / (1 + np.exp(-x)))
data = {
    'Hours Studied': [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50],
    'Pass': [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Create and train the logistic regression model
X = df[['Hours Studied']]
y = df['Pass']

logr_model = linear_model.LogisticRegression()
logr_model.fit(X, y)
X_wiki_list=[]
X_wiki_list=list(X['Hours Studied'])
y_lst=[]
y_lst=list(y)
y_graph=[]
# Find the intercept (beta0) and coefficient (beta1)
beta0 = logr_model.intercept_[0]
beta1 = logr_model.coef_[0][0]

# Plotting the sigmoid curve
for i in X_wiki_list:
    t=beta0+(i*beta1)
    y_graph.append(logistic_function(t))

# y_array = np.array(y_graph)

print(f'Intercept (beta0): {beta0:.4f}')
print(f'Coefficient (beta1): {beta1:.4f}')

plt.figure(figsize=(8, 6))
# plt.yticks(custom_y_ticks)
plt.plot(X_wiki_list,y_graph, label="Logistic Function", color='blue')
plt.xlabel("hours studied")
plt.ylabel("Possibility of Passing")
plt.title("Logistic Regression Graph for Wiki Pass/Fail")
plt.legend()
plt.grid(True)
plt.show()


