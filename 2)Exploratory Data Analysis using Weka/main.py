import pandas as pd

# Load dataset
data = pd.read_csv("loans.csv")  # Use your dataset path here
print(data.head())  # Preview the first few rows
print(data.info())  # Overview of dataset (data types, missing values)
print(data.describe())  # Statistical summary of numerical attributes

from sklearn.impute import SimpleImputer


# Impute missing numerical values with median
numerical_columns = ['loan_amount', 'rate']
imputer = SimpleImputer(strategy='median')
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])
# Impute missing date values with the most frequent value
date_columns = ['loan_start', 'loan_end']
date_imputer = SimpleImputer(strategy='most_frequent')
data[date_columns] = date_imputer.fit_transform(data[date_columns])
# Impute categorical column with most frequent value
categorical_columns = ['loan_type']
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])



# IQR method to filter outliers in loan_amount and rate
for col in ['loan_amount', 'rate']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]
    

#normalising the data

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data[['loan_amount', 'rate']] = scaler.fit_transform(data[['loan_amount', 'rate']])

data = pd.get_dummies(data, columns=['loan_type'], drop_first=True)
data['loan_amount_binned'] = pd.cut(data['loan_amount'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
print(data['loan_amount_binned'])

#concept hierarchy

# def loan_type_hierarchy(loan_type):
#     if loan_type in ["home_loan", "auto_loan"]:
#         return "secured_loan"
#     elif loan_type in ["personal_loan", "business_loan"]:
#         return "unsecured_loan"
#     else:
#         return "other"

# # Apply the hierarchy function to create a new column
# data['loan_type_hierarchy'] = data['loan_type'].apply(loan_type_hierarchy)

# #on loan amount
# def loan_amount_hierarchy(amount):
#     if amount < 50000:
#         return "low"
#     elif 50000 <= amount < 150000:
#         return "medium"
#     else:
#         return "high"

# # Apply the function
# data['loan_amount_hierarchy'] = data['loan_amount'].apply(loan_amount_hierarchy)


#pca
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Select only numerical columns to perform PCA
numerical_cols = ['loan_amount', 'rate']  # Add other numerical columns if available
data_numerical = data[numerical_cols]

# Step 1: Standardize the numerical data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numerical)

# Step 2: Apply PCA to reduce to 2 principal components
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Step 3: Add the PCA results back to the original dataset
data['PCA1'] = data_pca[:, 0]
data['PCA2'] = data_pca[:, 1]

# Print explained variance for each component
print("Explained variance by each component:", pca.explained_variance_ratio_)

# Step 4: Visualize the PCA components
plt.figure(figsize=(8, 6))
plt.scatter(data['PCA1'], data['PCA2'], c='blue', alpha=0.5)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA of Loan Data')
plt.show()


