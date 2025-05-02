#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv(r"C:\Users\Alisher Amer\Downloads\shopping_behavior_updated.csv")
print(df)


# In[3]:


#checking for missing values
MV = df.isnull().sum()
print(MV)


# In[4]:


#checking for outliers after inspection of the dataset (no outliers were seen but just in case)

Q1 = df['Purchase Amount (USD)'].quantile(0.25)
Q3 = df['Purchase Amount (USD)'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Purchase Amount (USD)'] < (Q1 - 1.5 * IQR)) | 
                         (df['Purchase Amount (USD)'] > (Q3 + 1.5 * IQR))]
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Age'] < (Q1 - 1.5 * IQR)) | 
                         (df['Age'] > (Q3 + 1.5 * IQR))]
print(outliers)


# In[7]:


from sklearn.preprocessing import LabelEncoder

# Initialize label encoders for each categorical column
encoders = {col: LabelEncoder() for col in df.select_dtypes(include=['object']).columns}

# Encode each of the columns
for col, encoder in encoders.items():
    df[col] = encoder.fit_transform(df[col])

# Display the new dataset
df.head()


# In[16]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Selecting relevant features for customer segmentation
features = ['Age', 'Gender', 'Category', 'Previous Purchases']
X = df[features]

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sum_of_squared_distances = []
K = range(1,11)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X_scaled)
    sum_of_squared_distances.append(km.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k (Number of clusters)')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[18]:


n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Adding the cluster information to the original dataset
df['Cluster'] = clusters

# Calculating the silhouette score to assess the quality of the clusters
silhouette_avg = silhouette_score(X_scaled, clusters)

print(silhouette_avg)


# In[17]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Plotting the PCA-reduced data
plt.figure(figsize=(10, 8))
plt.scatter(principal_components[:, 0], principal_components[:, 1], s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA plot of the Shopping Data')
plt.grid(True)
plt.show()

#We can see that points on the graph are dense in the middle


# In[15]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Calculating the average use of discounts and promo codes
avg_discount_use = df['Discount Applied'].mean()
avg_promo_use = df['Promo Code Used'].mean()

# Identifying customers who meet the criteria for 'Target Customer'
df['Target Customer'] = ((df['Discount Applied'] > avg_discount_use) &
                                    (df['Promo Code Used'] > avg_promo_use))
                                

# Selecting features for the model, excluding direct indicators of 'Target Customer'
features_for_target_customer = ['Age', 'Gender', 'Previous Purchases']
X_target_customer = df[features_for_target_customer]
y_target_customer = df['Target Customer']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_target_customer, y_target_customer, test_size=0.3, random_state=42)

# Building the Random Forest Classifier for 'Target Customer'
rf_classifier_tc = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_tc.fit(X_train, y_train)

# Making predictions on the test set for 'Target Customer'
y_pred = rf_classifier_tc.predict(X_test)

# Evaluating the model for 'Target Customer'
accuracy= accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display the model performance for 'Target Customer'
accuracy, classification_rep


# In[19]:


from sklearn.linear_model import LogisticRegression


# Features for the logistic regression model
features_for_subscription = ['Frequency of Purchases', 'Previous Purchases', 'Age', 'Gender', 'Review Rating', 
                            'Item Purchased', 'Category', 'Location', 'Discount Applied']

# Selecting features and target variable
X_subscription = df[features_for_subscription]
y_subscription = df['Subscription Status']

# Splitting the dataset into training and testing sets
X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_subscription, y_subscription, test_size=0.3, random_state=42)

# Building the Logistic Regression model
logreg_classifier_sub = LogisticRegression(max_iter=1000, random_state=42)
logreg_classifier_sub.fit(X_train_sub, y_train_sub)

# Making predictions on the test set
y_pred_sub = logreg_classifier_sub.predict(X_test_sub)

# Evaluating the model
accuracy_sub = accuracy_score(y_test_sub, y_pred_sub)
classification_rep_sub = classification_report(y_test_sub, y_pred_sub)

# Display the model performance
accuracy_sub, classification_rep_sub


# In[ ]:




