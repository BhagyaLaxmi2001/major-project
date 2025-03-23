import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
df=pd.read_csv('dont.csv')
df
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
# Assuming 'timestamp' is a valid column name in your DataFrame
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df = df.drop('timestamp', axis=1)
df
df=df.dropna()
df
missing_data = df.isnull().sum()
print(missing_data)
target_variables = ['act379', 'act13', 'act279', 'act323', 'act363', 'act302']
features = ['day_of_week','month','latitude', 'longitude']
X = df[features]
Y = df[target_variables]
random_state_value =1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=random_state_value)
clf_dt = DecisionTreeClassifier(random_state=random_state_value)
clf_dt.fit(X_train, Y_train)
Y_pred_dt = clf_dt.predict(X_test)
accuracy_dt = accuracy_score(Y_test, Y_pred_dt)
precision_dt = precision_score(Y_test, Y_pred_dt, average='weighted')
recall_dt = recall_score(Y_test, Y_pred_dt, average='weighted')
f1_dt = f1_score(Y_test, Y_pred_dt, average='weighted')
print("Decision Tree Classifier:")
print("Accuracy:", accuracy_dt)
print("Precision:", precision_dt)
print("Recall:", recall_dt)
print("F1 Score:", f1_dt)
importances = clf_dt.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importances - Decision Tree Classifier')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, Y_train)
Y_pred_rf = clf_rf.predict(X_test)
accuracy_rf = accuracy_score(Y_test, Y_pred_rf)
precision_rf = precision_score(Y_test, Y_pred_rf, average='weighted')
recall_rf = recall_score(Y_test, Y_pred_rf, average='weighted')
f1_rf = f1_score(Y_test, Y_pred_rf, average='weighted')
print("\nRandom Forest Classifier:")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1 Score:", f1_rf)
importances_rf = clf_rf.feature_importances_
feature_importance_df_rf = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances_rf})
feature_importance_df_rf = feature_importance_df_rf.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df_rf, palette='viridis')
plt.title('Feature Importances - Random Forest Classifier')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(X_train, Y_train)
Y_pred_knn = clf_knn.predict(X_test)
accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
precision_knn = precision_score(Y_test, Y_pred_knn, average='weighted')
recall_knn = recall_score(Y_test, Y_pred_knn, average='weighted')
f1_knn = f1_score(Y_test, Y_pred_knn, average='weighted')
print("\nk-Nearest Neighbors Classifier:")
print("Accuracy:", accuracy_knn)
print("Precision:", precision_knn)
print("Recall:", recall_knn)
print("F1 Score:", f1_knn)
algorithms = ['Decision Tree', 'Random Forest', 'k-Nearest Neighbors']
accuracies = [accuracy_dt, accuracy_rf, accuracy_knn]
plt.figure(figsize=(10, 6))
plt.bar(algorithms, accuracies, color=['blue', 'green', 'orange'])
plt.title('Accuracy Comparison of Different Algorithms')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set the y-axis limit to 0-1 for accuracy
plt.show()
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, Y_train)