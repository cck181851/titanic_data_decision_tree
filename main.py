# Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load the the from Titanic dataset, the dataset should be available in the same directory as the source code 
file_path = "train.csv"  
titanic_data = pd.read_csv(file_path) #reading the data from csv file

# the function that handles preprocessing step
def preprocess(data):
    #drop the columns that do not give useful information about the survival chance of a passenger
    data = data.drop(columns=[col for col in ["PassengerId", "Name", "Ticket", "Cabin"] if col in data.columns])
    
    # Handle missing values, fill with the median value
    data["Age"] = data["Age"].fillna(data["Age"].median())  
    if "Embarked" in data.columns:
        data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])  

    #convert the columns Sex and Embarked into numerical value  
    if "Sex" in data.columns:
        data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
    if "Embarked" in data.columns:
        data["Embarked"] = data["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    return data

# Preprocess the Titanic dataset, after the preprocessing, the data is ready to be used for the task
titanic_data = preprocess(titanic_data)


# Define features and target 
X = titanic_data.drop('Survived', axis=1) 
y = titanic_data['Survived'] 

# Split the dataset into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize the Decision Tree Classifier with entropy criterion and pruning
tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=10, random_state=42)

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store results of training and validation accuracies
train_accuracies = []
validation_accuracies = []

for fold, (train_index, val_index) in enumerate(kf.split(tree_classifier), start=1):
    X_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # Train the model on the training set
    tree_classifier.fit(X_train, y_train)

    # Predict on both the training and validation sets
    train_preds = tree_classifier.predict(X_train)
    val_preds = tree_classifier.predict(X_val)

    # Calculate the accuracies
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)

    train_accuracies.append(train_acc)
    validation_accuracies.append(val_acc)

# Calculate the error rates
train_errors = [1 - acc for acc in train_accuracies]
validation_errors = [1 - acc for acc in validation_accuracies]

# Train the decision tree using the entire training set
tree_classifier.fit(X_train, y_train)

# Evaluate the model on the test set
test_preds = tree_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, test_preds)
test_error = 1 - test_accuracy

# Plot combined error rates
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), train_errors, label='Training Error', marker='o')
plt.plot(range(1, 6), validation_errors, label='Validation Error', marker='x')
plt.axhline(y=test_error, color='r', linestyle='--', label='Test Error')  # Test error as a horizontal line
plt.xlabel('Fold')
plt.ylabel('Error Rate')
plt.title('Training, Validation, and Test Errors')
plt.legend()
plt.show()

# Visualize the final decision tree
plt.figure(figsize=(16, 12))
plot_tree(
    tree_classifier,
    feature_names=X_train.columns,
    class_names=['Not Survived', 'Survived'],
    filled=True,
    rounded=True
)
plt.title("Final Decision Tree")
plt.savefig("final_decision_tree.png", dpi=300)
plt.show()

# Plot feature importance
importances = tree_classifier.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(X_train.columns, importances)
plt.xticks(rotation=45)
plt.title('Feature Importance')
plt.show()
