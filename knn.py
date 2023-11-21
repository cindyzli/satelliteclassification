#knn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from skimage import color
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy

# loading from files made by run.py
with open('train_data.pkl', 'rb') as f:
    train_features, train_labels = pickle.load(f)

# split data
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels[0], test_size=0.2, random_state=42)

# knn training
def train_knn(X_train, y_train, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

# hyperparameter w gridsearch
def tune_knn(X_train, y_train):
    param_grid = {'n_neighbors': [3, 6, 8, 10, 13]}  #i tried varying k values but doesnt change much (within 1%)
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_['n_neighbors']

# feature engineering
def extract_features(images):
    extracted_features = []

    for i in range(images.shape[-1]):
        image = images[:, :, :, i]

        # exclude the alpha channel if needed
        if image.shape[2] == 4:
            image = image[:, :, :3]

        intensity_channel = color.rgb2gray(image)
        mean_intensity = np.mean(intensity_channel)
        std_intensity = np.std(intensity_channel)
        variance_intensity = np.var(intensity_channel)
        second_moment_intensity = np.mean(intensity_channel**2)
        entropy_intensity = -np.sum(intensity_channel * np.log(intensity_channel + 1e-10))

        if image.shape[2] > 3:
            mean_nir = np.mean(image[:, :, 3])
            std_nir = np.std(image[:, :, 3])
            variance_nir = np.var(image[:, :, 3])
            second_moment_nir = np.mean(image[:, :, 3]**2)
            entropy_nir = -np.sum(image[:, :, 3] * np.log(image[:, :, 3] + 1e-10))
        else:
            mean_nir, std_nir, variance_nir, second_moment_nir, entropy_nir = 0, 0, 0, 0, 0

        lbp = local_binary_pattern(intensity_channel, P=8, R=1, method='uniform')
        unique_values, counts = np.unique(lbp, return_counts=True)
        entropy_lbp = shannon_entropy(counts / counts.sum())

        extracted_features.append([mean_intensity, std_intensity, variance_intensity, second_moment_intensity, entropy_intensity,
                                   mean_nir, std_nir, variance_nir, second_moment_nir, entropy_nir,
                                   entropy_lbp])

    return np.array(extracted_features)

#train on a small set
k_value = tune_knn(X_train[:1000], y_train[:1000])

#visualize some of the data
plt.scatter(X_train[:1000, 0], X_train[:1000, 1], c=y_train[:1000], cmap='viridis')
plt.title('Visualization of Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#train with rest of set
knn_classifier = train_knn(X_train, y_train, k_value)

with open('knn_model.pkl', 'wb') as model_file:
    pickle.dump(knn_classifier, model_file)

#eval
val_predictions = knn_classifier.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
#~88% accuracy
