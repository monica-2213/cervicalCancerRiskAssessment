import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
nltk.download('punkt')
nltk.download('wordnet')

# Load the dataset from CSV
data = pd.read_csv("cervical_cancer.csv")

# Split the data into features and labels
X = data.drop('risk_level', axis=1)
y = data['risk_level']

# User input
user_input = input("Please describe your symptoms: ")

# Extract age from user input
age_pattern = r"(?i)(?:\b(\d+)\s*(?:years? old)|age\s*(\d+))"
age_match = re.search(age_pattern, user_input)
if age_match:
    if age_match.group(1) is not None:
        age = int(age_match.group(1))
    elif age_match.group(2) is not None:
        age = int(age_match.group(2))
    else:
        age = 0
else:
    age = 0

# Tokenize user input
tokens = word_tokenize(user_input.lower())

# Lemmatize user input tokens
lemmatizer = WordNetLemmatizer()
user_input_tokens_lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
print(user_input_tokens_lemmatized)

# Symptoms list
symptom_list = ['bleeding_between_periods', 'hurt_during_sex', 'pelvic_pain', 'abnormal_vaginal_discharge']


# Detect symptoms
undetected_symptoms = []
detected_symptoms = []
negation_words = ['not', 'no', 'n\'t', 'dont', 'don\'t', 'never']
context_window = 3

# Check for symptoms in user input
for symptom in symptom_list:
    symptom_tokens = symptom.split('_')
    symptom_lemmas = [lemmatizer.lemmatize(token) for token in symptom_tokens]

    # Check if any of the symptom lemmas exist in the user input
    for lemma in symptom_lemmas:
        if lemma in user_input_tokens_lemmatized:
            # Check for negation words within the context window around the symptom
            negation_found = False
            symptom_index = user_input_tokens_lemmatized.index(lemma)
            start_index = max(0, symptom_index - context_window)
            end_index = min(len(user_input_tokens_lemmatized), symptom_index + context_window + 1)
            context_tokens = user_input_tokens_lemmatized[start_index:end_index]

            for token in context_tokens:
                if token in negation_words:
                    negation_found = True
                    break

            if negation_found:
                undetected_symptoms.append(symptom)
            elif symptom not in detected_symptoms:
                detected_symptoms.append(symptom)

# Create a dictionary to map the column names to binary values
symptoms_mapping = {symptom: 1 for symptom in detected_symptoms}

# Create a new list to store the binary values
symptoms_values = [age]+[symptoms_mapping.get(symptom, 0) for symptom in symptom_list]

print("Detected symptoms:", detected_symptoms)
print("Age:", age)
print("Symptoms Value based on dataset symptoms :" ,symptoms_values)

#ini kalau nk guna ML
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Make predictions for the symptoms_values
prediction = classifier.predict([symptoms_values])

# Print the predicted risk level
print("Predicted risk level:", prediction[0])

# Evaluate the classifier on the test set
y_pred = classifier.predict(X_test)
# print(classification_report(y_test, y_pred))
