# Sales_prediction
The dataset used in this script was obtained from Kaggle, specifically from the following link: [https://www.kaggle.com/datasets/hunter0007/ecommerce-dataset-for-predictive-marketing-2023]. The script utilizes the Pandas library to load and explore the dataset with the objective of predicting the product a user is likely to purchase next based on their purchase history.

To handle missing values in the 'days_since_prior_order' column, NaN values were replaced with -1, and the data type was converted to integers. Subsequently, some parts of the data were visualized to gain insights into patterns. To make the dataset consist of only numerical values, the 'product_name' and 'department_name' columns were dropped. Feature engineering was then employed to compute user-specific features such as the average order size and the most frequently purchased product, as well as product-specific features like reorder rate and popularity.

XGBoost, a gradient boosting algorithm, was applied to the dataset after splitting it into training, test, and validation sets. The target column for prediction was 'product_id.' The script defines XGBoost model parameters and trains the model on the training data. It then evaluates the model's accuracy on the test data, prints a classification report, and generates a confusion matrix. Finally, the trained XGBoost model is saved to a file called 'xgboost_model.model.'

Additionally, the script includes a testing section where a function called 'predict_product' is defined. This function predicts a product for a given user based on their features. It utilizes the trained XGBoost model to make predictions for specific user IDs and maps the predicted product ID to the product name using the original data. The script returns the predicted product name.

The results for the two test cases provided are as follows:
1. 'Predicted Product for User 38: bakery desserts'
2. 'User 1 history not found'

It's worth noting that there is no User 1 in the dataset, which explains the second result.

The script appears to serve as a proof-of-concept for predicting user behavior in an e-commerce context rather than aiming for maximum accuracy. The classification report and accuracy results may not be representative of a fully optimized model, and it's possible that there are errors or unrealistic aspects in the current implementation. Further refinement and optimization of the model may be necessary for practical use in predictive marketing.
