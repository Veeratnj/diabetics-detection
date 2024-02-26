import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model  # Import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report
import joblib

from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predicted probabilities to class labels
    y_test_classes = np.argmax(y_test, axis=1)  # Convert true labels to class labels
    report = classification_report(y_test_classes, y_pred_classes)
    print(report)



def preprocess_image(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray_image = cv2.resize(gray_image, (64, 64))
  gray_image = gray_image.astype("float") / 255.0
  return gray_image

def data_preprocessing(dataset_dir, csv_file_path):
  images = []
  labels = []

  try:
    labels_df = pd.read_csv(csv_file_path)
    image_filenames = labels_df.iloc[:, 0]

    for index, image_filename in enumerate(image_filenames):
      image_path = os.path.join(dataset_dir, image_filename)

      if os.path.isfile(image_path):
        image = cv2.imread(image_path)
        gray_image = preprocess_image(image)
        images.append(gray_image)
        class_labels = labels_df.iloc[index, 1:].values.astype(np.float32)  # Ensure labels are float
        labels.append(class_labels)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels
  except Exception as e:
    print("Error reading CSV file:", e)

def model_training(images, labels):
  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

  # Convert one-hot encoded labels to integer labels (e.g., choosing "No_DR" class)
#   y_train = np.argmax(y_train, axis=1)
#   y_test = np.argmax(y_test, axis=1)

  # Define the CNN model
  model = Sequential()

  # Add convolutional layers
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))

  # Flatten the output of the convolutional layers
  model.add(Flatten())

  # Add a fully connected layer to predict the class labels
  model.add(Dense(128, activation='relu'))
  model.add(Dense(5, activation='softmax'))

  # Compile the model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  # Train the model
  model.fit(X_train, y_train, epochs=10)

  # Evaluate the model
  evaluate_model(X_test=X_test,y_test=y_test,model=model)
  # y_pred = model.predict(X_test)
  # report = classification_report(y_test, np.argmax(y_pred, axis=1))  # Use np.argmax to get integer labels
  # print(report)
  # Save the trained model to a file
  model.save('trained_cnn_model.h5')

  return 'report'

def detect_diabetes(input_image_path):
  # Load the trained CNN model
  model = load_model('trained_cnn_model.h5')

  # Preprocess the input image
  input_image = cv2.imread(input_image_path)
  gray_image = preprocess_image(input_image)

  # Reshape the image to match the model input shape
  gray_image = np.reshape(gray_image, (1, 64, 64, 1))

  # Pass the preprocessed image to the model to predict the class labels
  predicted_labels = model.predict(gray_image)

  # Return the predicted class labels
  return predicted_labels

# Example usage:
if __name__ == "__main__":
  # Define the paths to your dataset directory and CSV file
  dataset_dir = 'E:\pp\dataset'
  csv_file_path = 'E:\pp\dataset\_classes.csv'

  # Step 1: Data Preprocessing
  images, labels = data_preprocessing(dataset_dir, csv_file_path)

  # Step 4: Model Training
  classification_report = model_training(images, labels)

  print("Classification Report:\n", classification_report)

  # Step 5: Detect Diabetes (Example)
  input_image_path = 'hi.jpg'  # Replace with the path to your input image
  predicted_labels = detect_diabetes(input_image_path)

  print("Predicted Labels:", predicted_labels)