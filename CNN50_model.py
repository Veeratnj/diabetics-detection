from imp import load_module
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import joblib



def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predicted probabilities to class labels
    y_test_classes = np.argmax(y_test, axis=1)  # Convert true labels to class labels
    report = classification_report(y_test_classes, y_pred_classes)
    print(report)


def detect_diabetes(input_image_path):
    # Load the trained BiLSTM-MSRCP model
    model = load_module('trained_resnet50_model.h5')

    # Preprocess the input image
    input_image = cv2.imread(input_image_path)
    gray_image = preprocess_image(input_image)

    # Reshape the image to match the model input shape
    gray_image = np.reshape(gray_image, (1, 64, 64, 1))

    # Pass the preprocessed image to the model to predict the class labels
    predicted_labels = model.predict(gray_image)

    # Return the predicted class labels
    return predicted_labels

def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    preprocessed_image = np.array(resized_image) / 255.0  # Normalize the pixel values
    return preprocessed_image

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
                preprocessed_image = preprocess_image(image)
                images.append(preprocessed_image)
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

    # Load the pre-trained ResNet-50 model without the top (classification) layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Add your custom classification layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(5, activation='softmax')(x)  # Adjust the output units based on your specific task

    # Create the new model
    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10)
    evaluate_model(X_test=X_test,y_test=y_test,model=model)

    # Save the trained model to a file
    model.save('trained_resnet50_model.h5')

    return 'report'

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
