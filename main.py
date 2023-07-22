import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the original dataset
df = pd.read_csv('D:/Downloads/customer_churn.csv')

# Drop irrelevant columns
df = df.drop(['Names', 'Location', 'Company', 'Onboard_date'], axis=1)

# Convert categorical variables to numeric variables
df['Account_Manager'] = pd.Categorical(df['Account_Manager'])
df['Account_Manager'] = df['Account_Manager'].cat.codes

df['Churn'] = pd.Categorical(df['Churn'])
df['Churn'] = df['Churn'].cat.codes

# Train the model
X = df.drop(['Churn'], axis=1)
y = df['Churn']

model = RandomForestClassifier()
model.fit(X, y)

# Plot the feature importances
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.xlabel('Relative Importance')
sns.set_style('whitegrid')
plt.show()

# Load the new data
new_customers_df = pd.read_csv('D:/Downloads/new_customers.csv')

# Preprocess the new data (apply the same preprocessing as the original data)
new_customers_df = new_customers_df.drop(['Names', 'Location', 'Company', 'Onboard_date'], axis=1)
new_customers_df['Account_Manager'] = pd.Categorical(new_customers_df['Account_Manager'])
new_customers_df['Account_Manager'] = new_customers_df['Account_Manager'].cat.codes

# Use the pre-trained model to predict churn on the new data
new_X = new_customers_df
new_churn_predictions = model.predict(new_X)

# Analyze the predictions
new_customers_df['Churn_Prediction'] = new_churn_predictions

# Print the churn predictions for new customers
print(new_customers_df[['Name', 'Churn_Prediction']])

