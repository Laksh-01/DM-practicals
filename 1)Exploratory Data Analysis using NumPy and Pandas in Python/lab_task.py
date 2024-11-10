import pandas as pd
import matplotlib.pyplot as plt


# LAB TASK 1  Load a CSV file into a Pandas DataFrame. Display the first five rows, 
# check for missing values, and generate summary statistics for the numerical columns.
df = pd.read_csv('data.csv')
print("First five rows of the DataFrame:->")
print(df.head())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nSummary statistics for numerical columns:")
print(df.describe())

#LAB TASK 2 Identify missing values in the dataset and fill them with the median value of their respective columns.
missing_values = df.isnull().sum()
print("Missing values in each column before filling:")
print(missing_values)

# Compute the median for each column with missing values
medians = df.median()

# Fill missing values with the median of their respective columns
df_filled = df.fillna(medians)


# Verify that there are no missing values left
missing_values_after = df_filled.isnull().sum()
print("\nMissing values in each column after filling:")
print(missing_values_after)


# TASK 3
# Specify the column and the threshold value
column_name = 'Duration'  # Replace with your actual column name
threshold = 200  # Replace with your actual threshold value

# Filter the DataFrame based on the condition
filtered_df = df[df[column_name] > threshold]

# Display the first five rows of the filtered DataFrame
print("Filtered DataFrame (rows where column value > threshold):")
print(filtered_df.head())


#TASK 4
# Group the dataset by a categorical column and calculate the mean and standard deviation of numerical columns for each group. Display the results.

# Specify the categorical column to group by
categorical_column = 'Duration'  # Replace with your actual categorical column name

# Group by the categorical column and calculate mean and standard deviation
grouped_df = df.groupby(categorical_column).agg(['mean', 'std'])

# Display the results
print("Grouped DataFrame with mean and standard deviation for numerical columns:")
print(grouped_df)


#TASK 5 -> Merge two DataFrames on a common column and display the first five rows of the merged DataFrame.
# Let's split the DataFrame into two based on a criterion
df1 = df[df['Duration'] <= 60]
df2 = df[df['Duration'] > 60]

# For demonstration, let's merge df1 and df2 on a common column, here 'Pulse'
# Note: This is just an example; 'Pulse' might not be a good merge key for real use cases
merged_df = pd.merge(df1, df2, on='Pulse', suffixes=('_left', '_right'))

# Display the first five rows of the merged DataFrame
print(merged_df.head())




#TASK 6  -> Create a new column in the DataFrame based on a calculation involving existing columns. Display the first five rows of the updated DataFrame.

df['Calories_per_Minute'] = df['Calories'] / df['Duration']
print(df.head())

#TASK 7 -> Calculate and display the correlation matrix for the numerical columns in the dataset.
correlation_matrix = df.corr()
print(correlation_matrix)
#TASK 8 -> Generate a scatter plot to visualize the relationship between two numerical columns. Include appropriate labels and titles.

plt.figure(figsize=(10, 6))
plt.scatter(df['Duration'], df['Calories'], color='blue', alpha=0.7)

# Adding labels and title
plt.xlabel('Duration (minutes)')
plt.ylabel('Calories burned')
plt.title('Scatter Plot of Duration vs. Calories Burned')

# Show the plot
plt.grid(True)
plt.show()