from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import base64
from io import BytesIO
from flask import render_template
from flask import render_template_string
from sklearn.model_selection import GridSearchCV
from IPython.display import display
import folium
from folium.plugins import HeatMap


app = Flask(__name__, template_folder='templates')
CORS(app)

# A simple hardcoded username and password for demonstration purposes
USERNAME = 'admin'
PASSWORD = 'password'

# Variable to track whether the user is logged in
logged_in = False

# Load your trained model
df = pd.read_csv('c://users//polabhagyalaxmi//dont.csv')

#preprocessing steps(converting the attributes to numerical format and date time format)

df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

#dropping the extracted timestamp atrribute
df = df.drop('timestamp', axis=1)

#checking whether there are any null values and printing the dataset again
df = df.dropna()
df

#segregating target and feature variables
target_variables = ['act379', 'act13', 'act279', 'act323', 'act363', 'act302']
features = ['day_of_week', 'month', 'hour','minute', 'latitude', 'longitude']

X = df[features]
Y = df[target_variables]

#Model building and evaluation for knn algorithm
random_state_value =1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=random_state_value)

clf_knn = KNeighborsClassifier(n_neighbors=11)
clf_knn.fit(X_train, Y_train)
Y_pred_knn = clf_knn.predict(X_test)

accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
precision_knn = precision_score(Y_test, Y_pred_knn, average='weighted')
recall_knn = recall_score(Y_test, Y_pred_knn, average='weighted')
f1_knn = f1_score(Y_test, Y_pred_knn, average='weighted')
print("\nk-Nearest Neighbors Classifier:")
print("Accuracy:", accuracy_knn)
print("Precission:",precision_knn)
print("Recall:", recall_knn)
print("F1 Score:", f1_knn)

#confusion matrix for knn algorithm
cm_knn = confusion_matrix(Y_test.values.argmax(axis=1), Y_pred_knn.argmax(axis=1))
plt.figure(figsize=(10, 8))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", xticklabels=target_variables, yticklabels=target_variables)
plt.title("Confusion Matrix - k-Nearest Neighbors Classifier")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
#plt.show()



#naive bayes model evaluation
# Your data and target variables
target_columns = ['act379', 'act13', 'act279', 'act323', 'act363', 'act302']
features = ['day_of_week', 'month', 'hour', 'minute', 'latitude', 'longitude']
X = df[features]
Y = df[target_columns]

# Split the data into training and testing sets
random_state_value = 1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=random_state_value)

# Define a dictionary to store models for each target column
models = {}

# Train a Gaussian Naive Bayes model for each target column
for target_column in target_columns:
    clf_nb = GaussianNB()
    Y_train_target = Y_train[target_column]
    clf_nb.fit(X_train, Y_train_target)
    models[target_column] = clf_nb

# Predict on the test set for each target column
Y_pred_nb = {}
for target_column in target_columns:
    clf_nb = models[target_column]
    Y_pred_nb[target_column] = clf_nb.predict(X_test)

# Evaluate the performance for each target column
for target_column in target_columns:
    accuracy_nb = accuracy_score(Y_test[target_column], Y_pred_nb[target_column])
    precision_nb = precision_score(Y_test[target_column], Y_pred_nb[target_column], average='weighted')
    recall_nb = recall_score(Y_test[target_column], Y_pred_nb[target_column], average='weighted')
    f1_nb = f1_score(Y_test[target_column], Y_pred_nb[target_column], average='weighted')

    # Print the results for each target column
    #print(f"\nGaussian Naive Bayes Classifier for {target_column}:")
    #print("Accuracy:", accuracy_nb)
    #print("Precision:", precision_nb)
    #print("Recall:", recall_nb)
    #print("F1 Score:", f1_nb)
# Assuming you have individual accuracy scores for each target column
accuracy_nb_act379 =0.8259187620889749  # Replace with actual accuracy value
accuracy_nb_act13 = 0.9206963249516441    # Replace with actual accuracy value
accuracy_nb_act279 = 0.6634429400386848   # Replace with actual accuracy value
accuracy_nb_act323 = 0.6634429400386848  # Replace with actual accuracy value
accuracy_nb_act363 = 0.9535783365570599   # Replace with actual accuracy value
accuracy_nb_act302 =0.9961315280464217   # Replace with actual accuracy value

# Calculate overall accuracy
Naivebayes_accuracy = sum([accuracy_nb_act379, accuracy_nb_act13, accuracy_nb_act279, accuracy_nb_act323, accuracy_nb_act363, accuracy_nb_act302]) / 6
print("NaiveBayesaccuracy:", Naivebayes_accuracy)


# Plot feature importance based on mean values
plt.figure(figsize=(12, 8))
for target_column in target_columns:
    feature_importance = models[target_column].theta_[0]  # Mean values for each feature in class 0 (assuming binary classification)
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.subplot(2, 3, target_columns.index(target_column) + 1)
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(f'Feature Importance for {target_column}')

#plt.tight_layout()
#plt.show()



#decision tree accuracy
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'criterion': ['gini', 'entropy']
}
dt_classifier = DecisionTreeClassifier(random_state=random_state_value)
dt_classifier.fit(X_train, Y_train)
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5)
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
best_dt = grid_search.best_estimator_
Y_pred_best = best_dt.predict(X_test)
accuracy_best = accuracy_score(Y_test, Y_pred_best)
precision_best = precision_score(Y_test, Y_pred_best, average='weighted')
recall_best = recall_score(Y_test, Y_pred_best, average='weighted')
f1_best = f1_score(Y_test, Y_pred_best, average='weighted')
print("\n Decision tree Performance Metrics:")
print("Accuracy:", accuracy_best)
print("Precision:", precision_best)
print("Recall:", recall_best)
print("F1 Score:", f1_best)

#confusion matrix for decision tree
cm_best = confusion_matrix(Y_test.values.argmax(axis=1), Y_pred_best.argmax(axis=1))
plt.figure(figsize=(10, 8))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Blues", xticklabels=target_variables, yticklabels=target_variables)
plt.title("Confusion Matrix - Decision Tree Classifier")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
#plt.show()

#feature importance of decision tree
feature_importance = best_dt.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Decision Tree Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
#plt.show()



#displaying a graph demonstarting comparision of each algorithms accuracies
algorithms = ['Decision Tree', 'Naive bayes', 'k-Nearest Neighbors']
accuracies = [accuracy_best, Naivebayes_accuracy, accuracy_knn]
plt.figure(figsize=(10, 6))
plt.bar(algorithms, accuracies, color=['blue', 'green', 'orange'])
plt.title('Accuracy Comparison of Different Algorithms')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set the y-axis limit to 0-1 for accuracy
#plt.show()



# Route for rendering the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    global logged_in

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == USERNAME and password == PASSWORD:
            logged_in = True
            # Redirect to the index page after successful login
            return redirect(url_for('index'))
        else:
            return render_template('logins.html', message='Invalid credentials')

    return render_template('logins.html', message=None)


@app.route('/')
def home():
    return render_template('home.html')

# Route for rendering the HTML form
@app.route('/index')
def index():
    global logged_in

    if not logged_in:
        return redirect(url_for('login'))

    return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        data = request.get_json()

        # Create a DataFrame with the user input
        user_df = pd.DataFrame(data, index=[0])

        # Fill missing values with zeros or any other suitable approach
        user_df.fillna(0, inplace=True)

        # Make predictions using the trained models
        prediction_dt = {}
        for target_column in target_variables:
            prediction_dt[target_column] = dt_classifier.predict(user_df[features])

        prediction_knn = {}
        for target_column in target_variables:
            prediction_knn[target_column] = clf_knn.predict(user_df[features])

        prediction_nb = {}
        for target_column in target_variables:
            clf_nb = models[target_column]
            prediction_nb[target_column] = clf_nb.predict(user_df[features])

        # Convert the predictions to lists for JSON serialization
        prediction_list_dt = {target_column: prediction.tolist() for target_column, prediction in prediction_dt.items()}
        prediction_list_knn = {target_column: prediction.tolist() for target_column, prediction in prediction_knn.items()}
        prediction_list_nb = {target_column: prediction.tolist() for target_column, prediction in prediction_nb.items()}

        # Return a JSON response with the predictions
        return jsonify({
            'prediction_dt': prediction_list_dt,
            'prediction_knn': prediction_list_knn,
            'prediction_nb': prediction_list_nb,
        })

    except Exception as e:
        return jsonify({'error': str(e)})
    
    

    # Route for rendering the features page
@app.route('/features')
def displayFeatures():
    try:
        # Get feature importance from the trained Decision Tree model
        feature_importance = dt_classifier.feature_importances_

        # Create a DataFrame for plotting
        feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Plot the feature importance graph
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title('Decision Tree Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')

        # Save the plot to BytesIO buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Encode the image to base64 for HTML display
        img_str = "data:image/png;base64," + base64.b64encode(buffer.read()).decode()

        # Close the plot to prevent memory leaks
        plt.close()

        # Return the image as a response
        return render_template('features.html', img_str=img_str)

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/display_charts', methods=['GET'])
def display_charts():
    chart_type = request.args.get('chart_type')

    if chart_type == 'day_of_week':
        img_str = "/static/pies_chart.png"  # URL for the static image
        return render_template('day_of_week_.html', img_str=img_str)

    elif chart_type == 'accuracies':
        acc_img_str = "/static/bar.png"  # URL for the static accuracy bar graph image
        return render_template('accuracie.html', acc_img_str=acc_img_str)
    
    elif chart_type == 'crime_density_map':  # Corrected value to match the button value
        den_img_str = "/static/density.png"  # URL for the static density map image
        return render_template('comparisions.html', img_url=den_img_str)  # Corrected variable name

    # Handle other cases or provide a default response
    return "Invalid chart type"

    

   # Route for rendering the predictions page
@app.route('/predictions')
def predictions():
    global logged_in

    if not logged_in:
        return redirect(url_for('login'))

    return render_template('predictions.html')


   



if __name__ == '__main__':
    app.run(debug=True)
