import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Load your dataset
training_dataset = pd.read_csv('Training.csv')
test_dataset = pd.read_csv('Testing.csv')

# Slicing and Dicing the dataset to separate features from predictions
X = training_dataset.iloc[:, 0:132].values
y = training_dataset.iloc[:, -1].values

# Dimensionality Reduction for removing redundancies
dimensionality_reduction = training_dataset.groupby(training_dataset['prognosis']).max()

# Encoding String values to integer constants
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# Splitting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Implementing the Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Saving the information of columns
cols = training_dataset.columns
cols = cols[:-1]

# Checking the Important features
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

# Load the custom dataset for diseases and symptoms
def load_disease_dataset(file_path):
    return pd.read_csv(file_path)

# Basic symptom checker function using custom dataset
def diagnose(symptoms, dataset):
    possible_diseases = []

    for _, row in dataset.iterrows():
        disease = row['disease']
        disease_symptoms = row['symptoms'].split(',')  # Split symptoms by comma
        disease_symptoms = [s.strip().lower() for s in disease_symptoms]  # Clean the symptoms

        # Find common symptoms between user input and disease's symptoms
        common_symptoms = set(disease_symptoms).intersection(set(symptoms))
        if common_symptoms:
            possible_diseases.append((disease, len(common_symptoms)))

    # Sort diseases by the number of matching symptoms
    possible_diseases.sort(key=lambda x: x[1], reverse=True)

    # If no diseases match, return a generic message
    if not possible_diseases:
        return "No specific disease matched. Consider consulting a doctor."

    # Return the top matched disease
    return f"Possible disease: {possible_diseases[0][0]} based on symptoms."

# GUI setup
class SymptomCheckerApp:
    def __init__(self, root, dataset):
        self.root = root
        self.dataset = dataset  # Store the dataset
        self.symptoms = []  # List to hold user-inputted symptoms
        self.current_symptom = 0  # Track the current symptom to ask
        
        self.label = tk.Label(root, text="Please respond with Yes/No to the symptoms:")
        self.label.pack()

        self.question_label = tk.Label(root, text="")
        self.question_label.pack()

        self.yes_button = tk.Button(root, text="Yes", command=self.yes_response)
        self.yes_button.pack()

        self.no_button = tk.Button(root, text="No", command=self.no_response)
        self.no_button.pack()

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_input)
        self.clear_button.pack()

        self.display_next_symptom()

    def display_next_symptom(self):
        # Display the next symptom to ask
        if self.current_symptom < len(cols):
            self.question_label.config(text=f"Do you have {cols[self.current_symptom]}?")
        else:
            self.show_diagnosis()

    def yes_response(self):
        # Record "Yes" response for the current symptom
        self.symptoms.append(cols[self.current_symptom])
        self.current_symptom += 1
        self.display_next_symptom()

    def no_response(self):
        # Record "No" response for the current symptom
        self.current_symptom += 1
        self.display_next_symptom()

    def show_diagnosis(self):
        # Call the diagnose function with the collected symptoms
        diagnosis = diagnose(self.symptoms, self.dataset)
        messagebox.showinfo("Diagnosis Result", diagnosis)

    def clear_input(self):
        # Clear all user responses
        self.symptoms = []
        self.current_symptom = 0
        self.display_next_symptom()


# Load the custom dataset (CSV)
doc_dataset = load_disease_dataset('doctors_dataset.csv')

diseases = dimensionality_reduction.index
diseases = pd.DataFrame(diseases)

doctors = pd.DataFrame()
doctors['name'] = np.nan
doctors['link'] = np.nan
doctors['disease'] = np.nan

doctors['disease'] = diseases['prognosis']
doctors['name'] = doc_dataset['Name']
doctors['link'] = doc_dataset['Description']

record = doctors[doctors['disease'] == 'AIDS']
record['name']
record['link']

# Execute the GUI
root = tk.Tk()
app = SymptomCheckerApp(root, doc_dataset)
root.mainloop()
