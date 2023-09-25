#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


df = pd.read_csv('ECommerce_consumer behaviour.csv')

df.head()

df.describe()

unique_values = df['user_id'].unique()

# Calculate the total number of unique values
total_unique_count = len(unique_values)

# Print the total number of unique values
print("Total number of unique values of user_id:", total_unique_count)


df.shape

# Use the 'value_counts' method to count the occurrences of each user_id
value_counts = df['user_id'].value_counts()

# Print the 10 most occurring user IDs
print("10 most occurring user IDs:")
print(value_counts.head(10))

# Print the 10 least occurring user IDs
print("\n10 least occurring user IDs:")
print(value_counts.tail(10))

missingV = df.isnull().sum().sort_values(ascending=False)
missingV = pd.DataFrame(data=df.isnull().sum().sort_values(ascending=False))
print(missingV)

df['days_since_prior_order'].unique()


# Replacing the 'nan' values
for col in df.columns:
    if df[col].dtypes == 'float64':
        df[col].fillna(-1,inplace=True) 

df.head()

df['days_since_prior_order']= df['days_since_prior_order'].astype(np.int64)

df.nunique()

df.info()

grouped = df.groupby("order_id")["order_dow"].aggregate("max").reset_index()
grouped = grouped.order_dow.value_counts()

sns.set_style('darkgrid')
sns.set_palette("rocket_r")
f, ax = plt.subplots(figsize=(17, 15))
sns.barplot(x=grouped.index, y=grouped.values, ax=ax)
ax.grid(True, axis='y')
plt.xticks(rotation='vertical', fontsize=15)
plt.yticks(fontsize=12)
plt.ylabel('Number of Orders', fontsize=15)
plt.xlabel('Day of The Week', fontsize=15)
plt.xticks([0,1, 2, 3, 4, 5, 6],["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
plt.title('Number of Orders Made Each Day', fontsize=20)
plt.show()

# Create a pivot table to aggregate the data
pivot_table = df.pivot_table(index='order_dow', columns='order_hour_of_day', values='order_id', aggfunc='count')

# Create a heatmap without annotations inside the cells
plt.figure(figsize=(10, 8))
sns.set(font_scale=1)
sns.heatmap(pivot_table, cmap='Reds', annot=False, fmt='d', linewidths=0.5)
plt.title('Number of Orders by Day of the Week and Hour of the Day', fontsize=15)
plt.xlabel('Hour of the Day')
plt.ylabel('Day of the Week')
plt.yticks([0,1, 2, 3, 4, 5, 6],["Sunday","Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
plt.xticks(range(24), ["12am", "1am", "2am", "3am", "4am", "5am", "6am", "7am", "8am", "9am", "10am", "11am", "12pm", "1pm", "2pm", "3pm", "4pm", "5pm", "6pm", "7pm", "8pm", "9pm", "10pm", "11pm"])
plt.xticks(rotation=45)
plt.show()

df = df.drop(['product_name', 'department'], axis=1)
df.head()

df.columns

user_avg_order_size = df.groupby('user_id')['add_to_cart_order'].mean().reset_index()
user_avg_order_size.rename(columns={'add_to_cart_order': 'avg_order_size'}, inplace=True)

user_avg_order_size


user_most_frequent_products = df.groupby('user_id')['product_id'].agg(lambda x: x.mode().iloc[0]).reset_index()
user_most_frequent_products.rename(columns={'product_id': 'most_frequent_product'}, inplace=True)

user_most_frequent_products

product_reorder_stats = df.groupby('product_id')['reordered'].agg(['count', 'sum']).reset_index()
product_reorder_stats['reorder_rate'] = product_reorder_stats['sum'] / product_reorder_stats['count']
product_reorder_stats.drop(['count', 'sum'], axis=1, inplace=True)

product_popularity = df['product_id'].value_counts().reset_index()
product_popularity.columns = ['product_id', 'popularity']

product_popularity


# Merge user-specific features into df based on 'user_id'
df = df.merge(user_avg_order_size, on='user_id', how='left')
df = df.merge(user_most_frequent_products, on='user_id', how='left')

# Merge product-specific features into df based on 'product_id'
df = df.merge(product_reorder_stats, on='product_id', how='left')
df = df.merge(product_popularity, on='product_id', how='left')

df.head()

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Define your X (features) and y (target) matrices
X = df[['user_id', 'order_number', 'order_dow', 'order_hour_of_day',
        'days_since_prior_order', 'add_to_cart_order',
        'avg_order_size', 'most_frequent_product',
        'reorder_rate', 'popularity']]

y = df['product_id']

y = y - 1

# Verify the shapes of the target and input feature datasets
print("Shape of y (target):", y.shape)
print("Shape of X (input features):", X.shape)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Further split the train set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Verify the shapes of the train, validation, and test sets
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_val:", y_val.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

# Create and train the XGBoost model
params = {
    'objective': 'multi:softmax',  # Use softmax for multiclass classification
    'num_class': len(y.unique()),  # Number of unique products
    'max_depth': 5,  
    'learning_rate': 0.005,  
    'n_estimators': 200, 
    'eval_metric': 'mlogloss',  # Set eval metric here
    'early_stopping_rounds': 10
}


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)],  verbose=1)

# Extract feature importances
feature_importances = model.feature_importances_

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print more detailed metrics
classification_rep = classification_report(y_test, y_pred)
confusion_mtx = confusion_matrix(y_test, y_pred)

print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", confusion_mtx)

model.save_model('xgboost_model.model')

model = xgb.XGBClassifier()
model.load_model('xgboost_model.model')
data = pd.read_csv('ECommerce_consumer behaviour.csv') 
def predict_product(user_id):
    if user_id not in df['user_id'].values:
        return "User not found"  
    
    # Retrieve the user's details from the DataFrame based on user_id
    user_details = df[df['user_id'] == user_id].iloc[0]

    # Extract relevant features for prediction
    user_inputs = [
        user_details['user_id'],
        user_details['order_number'],
        user_details['order_dow'],
        user_details['order_hour_of_day'],
        user_details['days_since_prior_order'],
        user_details['add_to_cart_order'],
        user_details['avg_order_size'],
        user_details['most_frequent_product'],
        user_details['reorder_rate'],
        user_details['popularity']
    ]
    # Make predictions using your XGBoost model
    predicted_product_id = model.predict([user_inputs])[0]

    # Map the product ID to the product name
    product_name = data[data['product_id'] == predicted_product_id]['product_name'].iloc[0]

    # Return the predicted product name
    return product_name

# Example usage
user_id_to_predict = 38  

# Check if the user_id exists before making predictions
if user_id_to_predict in df['user_id'].values:
    predicted_product_name = predict_product(user_id_to_predict)
    print(f"Predicted Product for User {user_id_to_predict}: {predicted_product_name}")
else:
    print(f"User {user_id_to_predict} not found in the DataFrame")
   
user_id_to_predict2 = 1

# Check if the user_id exists before making predictions
if user_id_to_predict2 in df['user_id'].values:
    predicted_product = predict_product(user_id_to_predict2)
    print(f"Predicted Product for User {user_id_to_predict2}: {predicted_product}")
else:
    print(f"User {user_id_to_predict2} not found in the DataFrame")





