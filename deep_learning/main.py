import text_vectorizing
import text_preprocessing
import text_cleaning
import data_importing
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Importing Data
print("Started Importing Data")

training_data_path = "../data/training_data.csv"
test_data_path = "../data/test_data.csv"
training_data, test_data = data_importing.data_importer(
    training_data_path, test_data_path)
del training_data_path, test_data_path
print("Importing done!!")


# Cleaning Text
print("\n\nStarted Cleaning Data")

training_data.insert(loc=2, column='cleaned', value=training_data['comment_text'].apply(
    lambda x: text_cleaning.text_cleaner(x)))
test_data.insert(loc=2, column='cleaned', value=test_data['comment_text'].apply(
    lambda x: text_cleaning.text_cleaner(x)))
print("  -Added Column Cleaned in Training and Testing Data")
print("Cleaning Done!!")


# Preprocessing Text
print("\n\nStarted Preprocessing Data")

print("Training Data:")
training_data = text_preprocessing.text_preprocessor(training_data)
print("Testing Data:")
test_data = text_preprocessing.text_preprocessor(test_data)

print(f'  -Columns in Training Data: {training_data.columns.values}')
print(f'  -Columns in Testing Data: {test_data.columns.values}')
print("Preprocessing Done!!")
print(training_data.shape)


# Creating New CSV
training_data.to_csv("logs/preprocessed_training_data.csv", index=False)
test_data.to_csv("logs/preprocessed_test_data.csv", index=False)
del training_data

print("\n\nPreprocessed Training/Test Data Saved to logs/")


# Train Test Split
print("\n\nSplitting Training Data into Train set and Val set in 4:1 Ratio.")

training_data = pd.read_csv("logs/preprocessed_training_data.csv")
X_train, X_val, y_train, y_val = train_test_split(
    training_data.iloc[:, :-7], training_data.iloc[:, -7:], test_size=0.2, random_state=42, stratify=(training_data["no_toxicity"].values))
print(f"  -Shape of X_train: {X_train.shape}")
print(f"  -Shape of X_val: {X_val.shape}")
print("Splitting Done!!")


# Vectorization
print("\n\nStarted Vectorization")
vector_dict_path = "logs/vectorized_data.pkl"
text_vectorizing.vectorize(X_train, X_val, vector_dict_path)
del X_train, X_val
print("Vectorization Done!!")