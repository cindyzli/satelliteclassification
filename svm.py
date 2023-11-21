#svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from skimage import color
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy

# svm
def train_svm(X_train, y_train, C, gamma):
    svm_classifier = SVC(C=C, gamma=gamma, kernel='rbf')
    svm_classifier.fit(X_train, y_train)
    return svm_classifier

# hyperparameter tuning w gridsearch
def tune_svm(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

# visualization
def visualize_svm(X_train, y_train, svm_classifier):

    h = .02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    features = extract_features(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])

    # replace nan values with zero (had problems without this)
    features = np.nan_to_num(features)

    Z = svm_classifier.predict(features)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title('SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    #only visualized some features

    plt.show()


# feature engineering
def extract_features(images):
    # Initialize an empty array to store extracted features
    extracted_features = []

    for i in range(images.shape[0]):
        image = images[i]

        intensity_channel = color.rgb2gray(image.reshape((1, 1, 3)))

        mean_intensity = np.mean(intensity_channel)
        std_intensity = np.std(intensity_channel)
        variance_intensity = np.var(intensity_channel)
        second_moment_intensity = np.mean(intensity_channel**2)
        entropy_intensity = -np.sum(intensity_channel * np.log(intensity_channel + 1e-10))

        # default vals
        mean_nir, std_nir, variance_nir, second_moment_nir, entropy_nir = 0, 0, 0, 0, 0

        # local binary pattern
        lbp = local_binary_pattern(intensity_channel, P=8, R=1, method='uniform')
        unique_values, counts = np.unique(lbp, return_counts=True)
        entropy_lbp = shannon_entropy(counts / counts.sum())

        extracted_features.append([mean_intensity, std_intensity, variance_intensity, second_moment_intensity, entropy_intensity,
                                   mean_nir, std_nir, variance_nir, second_moment_nir, entropy_nir,
                                   entropy_lbp])

    return np.array(extracted_features)

# normalize
def normalize_features(features):
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)

# serialize
def serialize_svm_model(model, filename):
    with open(filename, 'wb') as model_file:
        pickle.dump(model, model_file)

def main():
    # made from run.py
    with open('train_data.pkl', 'rb') as f:
        train_features, train_labels = pickle.load(f)

    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels[0], test_size=0.2, random_state=42)

    X_train_normalized = normalize_features(X_train)
    X_val_normalized = normalize_features(X_val)

    #train on small set
    svm_params = tune_svm(X_train_normalized[:1000], y_train[:1000])

    # visualize
    visualize_svm(X_train_normalized[:1000], y_train[:1000], train_svm(X_train_normalized[:1000], y_train[:1000], **svm_params))

    # train with rest of the set
    svm_classifier = train_svm(X_train_normalized, y_train, **svm_params)
 
    serialize_svm_model(svm_classifier, 'svm_model.pkl')

    # eval
    val_predictions_svm = svm_classifier.predict(X_val_normalized)
    val_accuracy_svm = accuracy_score(y_val, val_predictions_svm)
    print(f'SVM Validation Accuracy: {val_accuracy_svm * 100:.2f}%')
#~92% accuracy

if __name__ == "__main__":
    main()
