import h5py
import numpy as np
import matplotlib.pyplot as plt

def global_contrast_enhancement(image):
    # Apply global contrast enhancement using histogram equalization
    hist, bins = np.histogram(image.flatten(), 256, density=True)
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255  # Normalize to [0, 255]
    enhanced_image = np.interp(image.flatten(), bins[:-1], cdf).reshape(image.shape)
    return enhanced_image

def dct_local_detail_enhancement(image):
    # Apply Discrete Cosine Transform (DCT) for local detail enhancement
    dct_image = np.fft.fft2(image)
    dct_image[1:, 1:] *= 2  # Amplify high-frequency components
    enhanced_image = np.fft.ifft2(dct_image).real
    return enhanced_image

# Load HDF5 file
with h5py.File('C:/Users/Dell/OneDrive/Desktop/hd5/3RIMG_06APR2023_2315_L1C_ASIA_MER_V01R00.hdf5', 'r') as file:
    image_data = file['IMG_TIR'][:]  # Assuming the dataset is named 'data'

# Remove the extra dimension if present
if len(image_data.shape) == 3 and image_data.shape[0] == 1:
    image_data = image_data[0]

# Rescale the pixel values to [0, 255]
image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255

# Apply contrast enhancement
enhanced_image = global_contrast_enhancement(image_data)
enhanced_image = dct_local_detail_enhancement(enhanced_image)

# Plot original and enhanced images in 2D
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_data, cmap='gray', aspect='equal', vmin=0, vmax=255)  # Set intensity range to [0, 255]
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(enhanced_image, cmap='gray', aspect='equal', vmin=0, vmax=255)  # Set intensity range to [0, 255]
plt.title('Enhanced Image')
plt.axis('off')

plt.tight_layout()
plt.show()
