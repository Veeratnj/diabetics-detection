import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report
import joblib

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
    gray_image = (gray_image - gray_image.mean()) / gray_image.std()  # Normalize
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
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Define and compile the model
    model = Sequential()
    # Add convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    # Flatten the output of the convolutional layers
    model.add(Flatten())
    # Add fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Implement early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, epochs=30, validation_split=0.2, callbacks=[early_stopping])

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    evaluate_model(model=model,X_test=X_test,y_test=y_test)
    # Save the trained model
    model.save('trained_BiLSTM.h5')

    return history

def detect_diabetes(input_image_path):
    # Load the trained BiLSTM-MSRCP model
    model = load_model('trained_BiLSTM.h5')

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
    history = model_training(images, labels)

    print("Training History:", history)

    # Step 5: Detect Diabetes (Example)
    input_image_path = 'hi.jpg'  # Replace with the path to your input image
    predicted_labels = detect_diabetes(input_image_path)

    print("Predicted Labels:", predicted_labels)
