import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target
print(data)


scaler = StandardScaler()
features = scaler.fit_transform(data.iloc[:, :-1]) 
X = features
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
clf = DecisionTreeClassifier(random_state=42)

cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')

average_accuracy = np.mean(cv_scores)
std_deviation = np.std(cv_scores)

print("Cross-Validation Accuracy Scores:", cv_scores)
print("Average Accuracy:", average_accuracy)
print("Standard Deviation:", std_deviation)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
clf = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)

y_pred = best_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Best Parameters from Grid Search:", best_params)
print("Best Cross-Validation Accuracy:", best_score)
print("\nTest Set Accuracy:", accuracy)
print("\nClassification Report on Test Set:\n", report)
print("\nConfusion Matrix on Test Set:\n", conf_matrix)
