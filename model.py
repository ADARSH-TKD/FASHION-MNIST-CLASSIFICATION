# <--------------   IMPORTING LIBRARY   ------------------->
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras

# <----------------------------  DATA PREPROCESSING   ---------------------------------->
# LOADING DATA FROM KERAS LIB
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Expanding dimensions to match the expected input shape
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# FEATURE SCALING 
x_train = x_train / 255
x_test = x_test / 255

# SPLIT DATASET
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=2020)

# <------------------------------------    CONVOLUTIONAL     -------------------------------------------->
# CONVOLUTIONAL NEURAL NETWORK - MODEL BUILDING
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=2, strides=(1, 1), padding="valid", activation="relu", input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation="relu"),  # Hidden layer
    keras.layers.Dense(units=10, activation="softmax"),  # Output layer (10 classes)
])

# Print the model architecture
print(model.summary())

# COMPILING OUR MODEL
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
 
# TRAINING THE MODEL 
model.fit(x_train, y_train, epochs=10, batch_size=512, verbose=1, validation_data=(x_val, y_val))

# TESTING THE MODEL
m = model.predict(np.expand_dims(x_test[0], axis=0)).round(2)
print("Predicted Probabilities:", m)
predicted_class = np.argmax(m)
actual_class = y_test[0]
print(f"Predicted Class: {predicted_class}")
print(f"Actual Class: {actual_class}")


# Print all test labels for verification
print("Rechecking Test Labels:", y_test)

# EVALUATING THE MODEL
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4%}")

# VISUALIZING THE OUTPUT
plt.figure(figsize=(16,30))

j = 1
for i in np.random.randint(0, len(x_test), 30):
    plt.subplot(10, 3, j)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    # Reduce the font size of the title
    plt.title(f"Pred: {np.argmax(model.predict(np.expand_dims(x_test[i], axis=0)))}\nActual: {y_test[i]}", fontsize=8)
    plt.axis('off')
    j += 1

plt.tight_layout()  # Adjust spacing to prevent overlap
plt.show()


# CONFUSION MATRIX
# Import necessary libraries
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generating predictions for the test set
y_pred = model.predict(x_test)
y_pred_labels = [np.argmax(label) for label in y_pred]

# Define class labels (for Fashion MNIST)
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)

# Plotting the confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

from sklearn.metrics import classification_report
cr= classification_report(y_test, y_pred_labels, target_names=class_labels)
print(cr)