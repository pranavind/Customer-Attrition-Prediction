import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\ml project\\dataset1.csv")
print(df)
#Remove duplicate values from the data set\
df = df.drop_duplicates()
df
#Use the info() method to obtain a brief description of the DataFrame that includes details about the non-null values and data types.\
df.info
#To provide a statistically informed view of a dataset, it's helpful to know what type of data you're working with.\
df.describe()
#printing the data type of all the column\
column_data_types = df.dtypes
print(column_data_types)
##print unique values in each colum
unique_values_per_column = {}
for column in df.columns:
    unique_values = df[column].unique()
    unique_values_per_column[column] = unique_values

#print unique values in each colum
unique_values_per_column = {}
for column in df.columns:
    unique_values = df[column].unique()
    unique_values_per_column[column] = unique_values

# Print or access unique values for each column
for column, values in unique_values_per_column.items():
    print(f"Unique values in {column}: {values}")
print("\n")
#check the number of null values or missing values in each column\
missing_values = df.isna().sum()
print(missing_values)

print(df)

#Now to find the percentage customers who have churn
total_churned = df.loc[df['Churn'] == 'Yes'].shape[0]
total_customers = df.shape[0]
churn_rate = (total_churned / total_customers) * 100
print("Churn rate =", churn_rate)

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame
sns.countplot(data=df, x='Churn', hue='Churn')

# Show the plot
plt.show()

churn_counts = df['Churn'].value_counts()
print(churn_counts)

df['Customer Churn'] = df['Churn'].map({ 'No':0, 'Yes':1 })
print(df)


# categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
print("categorical_columns")
print(categorical_columns)
print("\n")
#numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns
print("numeric_columns")
print(numeric_columns)


#Here we are creating a new column for tenure in the name of tenure time here instaed of the numbers in months given in the data set we are grouping the data into duration
#'0-12 Months' '12-24 Months' '24-48 Months' 'Over 48 Months'

print(df.columns)


def tenure_duration(tenure):
    if tenure < 13:
        return '0-12 Months '
    elif tenure < 25:
        return '12-24 Months '
    elif tenure < 49:
        return '24-48 Months '
    else:
        return 'Over 48 Months '

df['tenure time'] = df['tenure'].apply(tenure_duration)
print(df)

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'tenure time' is the correct column name after correction
plt.figure(figsize=(10, 4), dpi=150)
sns.countplot(data=df, x='tenure time', hue='Customer Churn')
plt.show()

#encoding
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create a copy of the DataFrame to avoid modifying the original data
df1 = df.copy()
# Define a list of categorical column names
categorical_columns = [
    'customerID', 'gender', 'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges','tenure time'
]

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Assuming df is your DataFrame containing the data
# Initialize a label encoder
label_encoder = LabelEncoder()

# Preprocess each categorical column
for column in categorical_columns:
    # Fill NaN values with a placeholder string ('missing')
    df1[column] = df1[column].fillna('missing').astype(str)
    # Encode the column
    df1[column] = label_encoder.fit_transform(df1[column])


print(df1)



# Extract features (X) and target variable (y)
X = df1[['gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges']]  # Features
y = df1['Churn']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Print feature importances
imp_fe = []
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
imp_fe = feature_importances.tolist()
print("Feature Importances:")
print(feature_importances)

#calculation
'''
Percentage Importance for Feature

X=( Importance Score of Feature X/sum of importance Score of Feature X )×100

This normalization ensures that the sum of all feature importances adds up to 100%.

For example, if a feature has an importance score of 0.05 and the sum of all importance scores is 0.5, the percentage importance of that feature would be 0.05 0.5 × 100

= 10 % 0.5 0.05​×100=10%.
'''
print("percentage")
print("\n")
list1 = []

# Assuming feature_importances is a DataFrame or Series with feature names as index and importances as values
sum_feature_importances = feature_importances.sum()

# Normalize the feature importances to percentages and store in imp_fe list
imp_fe = (feature_importances / sum_feature_importances * 100).tolist()

# Append the normalized importances to list1
list1.extend(zip(feature_importances.index, imp_fe))

# Sort list1 in descending order based on importances
list1_sorted = sorted(list1, key=lambda x: x[1], reverse=True)

# Print the sorted elements with feature names
for feature, sorted_importance in list1_sorted:
   print(f"{feature}: {sorted_importance:.2f}%")


import matplotlib.pyplot as plt

# Assuming 'feature_importances' is the series containing importances

# Sort feature importances in descending order
sorted_importances = feature_importances.sort_values(ascending=False)

# Create a bar plot
plt.figure(figsize=(12, 8))
sorted_importances.plot(kind='bar')
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.show()



# graph
import matplotlib.pyplot as plt
import seaborn as sns
# Selecting only the categorical columns
categorical_columns = [  'PhoneService',
       'MultipleLines'
] 

num_cols = len(categorical_columns)
fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(15, 5 * num_cols))

# Loop through each column and create a count plot
for i, column in enumerate(categorical_columns):
    sns.countplot(x=column, hue="Churn", data=df1, ax=axes[i])
    axes[i].set_title(f'Countplot for {column}')
    axes[i].set_xlabel('')  # Remove x-axis label for better organization

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Increase the vertical space between subplots
plt.show()

# Selecting only the categorical columns
categorical_columns1 = ['gender', 'SeniorCitizen', 'Partner','Dependents', ]

num_cols = len(categorical_columns1)
fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(15, 5 * num_cols))

# Loop through each column and create a count plot
for i, column in enumerate(categorical_columns1):
    sns.countplot(x=column, hue="Churn", data=df1, ax=axes[i])
    axes[i].set_title(f'Countplot for {column}')
    axes[i].set_xlabel('')  # Remove x-axis label for better organization

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Increase the vertical space between subplots
plt.show()


categorical_columns2 = ['InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport']

num_cols = len(categorical_columns2)
fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(15, 5 * num_cols))

# Loop through each column and create a count plot
for i, column in enumerate(categorical_columns2):
    sns.countplot(x=column, hue="Churn", data=df1, ax=axes[i])
    axes[i].set_title(f'Countplot for {column}')
    axes[i].set_xlabel('')  # Remove x-axis label for better organization

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Increase the vertical space between subplots
plt.show()


categorical_columns3 = [
        'PaperlessBilling', 'PaymentMethod']

num_cols = len(categorical_columns3)
fig, axes = plt.subplots(nrows=num_cols, ncols=1, figsize=(15, 5 * num_cols))

# Loop through each column and create a count plot
for i, column in enumerate(categorical_columns3):
    sns.countplot(x=column, hue="Churn", data=df1, ax=axes[i])
    axes[i].set_title(f'Countplot for {column}')
    axes[i].set_xlabel('')  # Remove x-axis label for better organization

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Increase the vertical space between subplots
plt.show()






import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create a copy of the DataFrame to avoid modifying the original data
df2 = df1.copy()

# Define a list of categorical column names
categorical_columns = ['Churn', 'tenure time']

# Initialize a label encoder
label_encoder = LabelEncoder()

# Preprocess each categorical column
for column in categorical_columns:
    # Fill NaN values with a placeholder string ('missing')
    df2[column] = df2[column].fillna('missing').astype(str)
    # Fit and transform the column (for train set)
    df2[column] = label_encoder.fit_transform(df2[column])

# Now, 'df2' contains the encoded values for your categorical columns
print(df2)
correlation_matrix = df2.corr()

# Print correlation with the target variable
print(correlation_matrix['Churn'].sort_values(ascending=False))

print("\n")
print(df2.columns)


print("\n")



#Model logistic regression
print("\nlogistic regression:")
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

X = df2[[
    'Contract',
    'MonthlyCharges',
    'OnlineSecurity',
    'InternetService',
    'SeniorCitizen',
    'tenure time',
    'TotalCharges',
    ]] # Features
y = df2['Churn']  # Target variable
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression(max_iter=1000, solver='liblinear')

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=10)
print("\n")
# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)

# Plot the confusion matrix (for illustration purposes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

'''
precision=TP/TP+FP

recall =TP/TP+FN

F1=2pr/p+r
'''

# RandomForestClassifier
print("\nRandom Forest Model:")
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

X = df2[[
    'Contract',
    'MonthlyCharges',
    'OnlineSecurity',
    'InternetService',
    'SeniorCitizen',
    'tenure time',
    'TotalCharges',
    ]]  # Features
y = df2['Churn']  # Target variable
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
# Initialize the random forest classifier
forest_model = RandomForestClassifier(random_state=42)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(forest_model, X_train, y_train, cv=5)
print("\n")
# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# Fit the random forest model to the training data
forest_model.fit(X_train, y_train)

# Make predictions using the random forest model
y_pred_forest = forest_model.predict(X_test)
# Evaluate the random forest model
accuracy_forest = accuracy_score(y_test, y_pred_forest)
conf_matrix_forest = confusion_matrix(y_test, y_pred_forest)
classification_rep_forest = classification_report(y_test, y_pred_forest)


print(f'Accuracy: {accuracy_forest}')
print('\nConfusion Matrix:')
print(conf_matrix_forest)
print('\nClassification Report:')
print(classification_rep_forest)

# Plot the confusion matrix for the random forest
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_forest, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=forest_model.classes_, yticklabels=forest_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()
