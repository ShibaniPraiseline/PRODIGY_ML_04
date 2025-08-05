import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def load_images(data_path):
    images = []
    labels = []
    for dir_name in os.listdir(data_path):
        label = dir_name
        for img_file in os.listdir(os.path.join(data_path, dir_name)):
            img_path = os.path.join(data_path, dir_name, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
            img = cv2.resize(img, (64, 64))  # resize for consistency
            images.append(img.flatten())
            labels.append(label)
    return np.array(images), np.array(labels)

# Load data
X, y = load_images(r"C:\path\to\your\hand_gesture_images")

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Visualize predictions
for i in range(5):
    img = X_test[i].reshape(64, 64)
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {y_pred[i]}, Actual: {y_test[i]}")
    plt.axis('off')
    plt.show()
