#Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
from skimage import io, color, feature
import numpy as np

# Load your image dataset (spam and non-spam images)
# Replace 'your_dataset_path' with the actual path to your dataset
spam_images = io.imread_collection('your_dataset_path/spam/*.jpg')
non_spam_images = io.imread_collection('your_dataset_path/non_spam/*.jpg')

# Extract features from the images using histogram of oriented gradients (HOG)
def extract_features(images):
    features = []
    for img in images:
        img_gray = color.rgb2gray(img)
        hog_features = feature.hog(img_gray, block_norm='L2-Hys', pixels_per_cell=(16, 16))
        features.append(hog_features)
    return np.array(features)

# Extract features for spam and non-spam images
spam_features = extract_features(spam_images)
non_spam_features = extract_features(non_spam_images)

# Create labels (1 for spam, 0 for non-spam)
spam_labels = np.ones(len(spam_features))
non_spam_labels = np.zeros(len(non_spam_features))

# Concatenate features and labels
X = np.concatenate((spam_features, non_spam_features), axis=0)
y = np.concatenate((spam_labels, non_spam_labels), axis=0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix)
