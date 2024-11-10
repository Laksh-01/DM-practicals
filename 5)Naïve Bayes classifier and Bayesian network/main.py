import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load and prepare the dataset
df = pd.read_csv('titanic.csv')
print("Initial DataFrame:")
print(df.head())

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Convert categorical columns to numerical values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Feature selection
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = df['Survived']  # Assuming this column exists

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Naive Bayes model: {accuracy:.2f}")

# Feature selection using SelectKBest with Chi-squared test
selector = SelectKBest(score_func=chi2, k='all')
selector.fit(X_train, y_train)

# Get scores and p-values
scores = selector.scores_
p_values = selector.pvalues_

# Create DataFrame to display feature scores and p-values
feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': scores, 'p-value': p_values})
print("\nFeature scores and p-values:")
print(feature_scores)

# Select features based on p-value < 0.05
selected_features = feature_scores[feature_scores['p-value'] < 0.05]['Feature']
print("\nSelected Features based on p-value < 0.05:")
print(selected_features.tolist())

# Cross-validation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
cv_scores = cross_val_score(model, X_scaled, y, cv=10, scoring='accuracy')

print("Cross-Validation Accuracy for each fold:")
print(cv_scores)
print(f"\nMean Accuracy: {cv_scores.mean():.2f}")

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)



#

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('titanic.csv')

# Define the features and target
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine X_train and y_train into a single DataFrame for fitting
train_data = X_train.copy()
train_data['Survived'] = y_train

# Manually construct the Bayesian Network with all relevant nodes and edges
model = BayesianNetwork([
    ('Pclass', 'Survived'),
    ('Sex', 'Survived'),
    ('Age', 'Survived'),
    ('Fare', 'Survived'),
    ('SibSp', 'Survived'),
    ('Parch', 'Survived')
])

# Track the nodes in the model
model_nodes = [node for node in model.nodes()]

# Function to check if nodes exist in the model
def check_nodes_in_model(evidence, model_nodes):
    """
    Checks if all keys in the evidence dictionary exist in the model's nodes.
    Filters out any evidence not present in model_nodes.
    """
    filtered_evidence = {key: value for key, value in evidence.items() if key in model_nodes}
    missing_nodes = [key for key in evidence if key not in model_nodes]
    if missing_nodes:
        print(f"Warning: The following nodes are missing in the model and will be ignored: {missing_nodes}")
    return filtered_evidence

# Fit the model with Maximum Likelihood Estimation
model.fit(train_data, estimator=MaximumLikelihoodEstimator)

# Create an inference object for querying the model
inference = VariableElimination(model)

# Initialize predictions list
y_pred_bn = []


# Initialize predictions list with default predictions
y_pred_bn = []

# Iterate through the test instances and make predictions
for _, row in X_test.iterrows():
    # Ensure only relevant evidence is passed to the model
    evidence = row[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']].to_dict()
    # Filter evidence to include only nodes that exist in the model and have no NaN values
    filtered_evidence = check_nodes_in_model(evidence, model_nodes)
    # Remove NaN values from the evidence dictionary
    filtered_evidence = {k: v for k, v in filtered_evidence.items() if pd.notna(v)}

    # Discretize continuous variables like 'Age' and 'Fare'
    for var in ['Age', 'Fare']:
        if var in filtered_evidence:
            discretized_value = pd.cut([filtered_evidence[var]], bins=5, labels=False)
            if discretized_value.size > 0 and pd.notna(discretized_value[0]):
                filtered_evidence[var] = int(discretized_value[0])  # Cast to int for consistency
            else:
                del filtered_evidence[var]
                print(f"Warning: Discretization for {var} resulted in an invalid label. Removing from evidence.")

    # Query the model
    try:
        result = inference.query(variables=['Survived'], evidence=filtered_evidence)
        predicted_class = 1 if result.values[1] > result.values[0] else 0
    except KeyError as e:
        predicted_class = 0  # Assign a default prediction (e.g., 'Not Survived')

    # Append the prediction to y_pred_bn
    y_pred_bn.append(predicted_class)

#Check for length consistency before evaluating
if len(y_pred_bn) == len(y_test):
    accuracy = accuracy_score(y_test, y_pred_bn)
    print(f"Bayesian Network Accuracy on Test Data: {accuracy}")

    # Generate confusion matrix if lengths are consistent
    cm = confusion_matrix(y_test, y_pred_bn)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Bayesian Network)')
    plt.show()
else:
    print("Warning: Length mismatch between y_test and y_pred_bn.")
    print(f"y_test length: {len(y_test)}, y_pred_bn length: {len(y_pred_bn)}")
