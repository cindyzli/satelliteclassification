import numpy as np
import pickle
from skimage import color
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
import scipy.io

try:
    data = scipy.io.loadmat('sat-4-full.mat')
except Exception as e:
    print(f"Error loading MATLAB file: {e}")
    raise

train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']

train_subset_size = 10000 
test_subset_size = 2500
train_x_subset = train_x[:, :, :, :train_subset_size]
train_y_subset = train_y[:, :train_subset_size]
test_x_subset = test_x[:, :, :, :test_subset_size]
test_y_subset = test_y[:, :test_subset_size]

# feature eng
def extract_features(images):
    extracted_features = []

    for i in range(images.shape[-1]):
        image = images[:, :, :, i]

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


train_features = extract_features(train_x_subset)
test_features = extract_features(test_x_subset)


with open('train_data.pkl', 'wb') as f:
    pickle.dump((train_features, train_y_subset), f)
    print("Train data saved.")

with open('test_data.pkl', 'wb') as f:
    pickle.dump((test_features, test_y_subset), f)
    print("Test data saved.")
