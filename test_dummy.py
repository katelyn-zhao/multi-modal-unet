import tensorflow as tf
print("tensorflow ver", tf.__version__)
import numpy as np
import nibabel as nib
import numpy as np
from skimage.transform import resize
from itertools import combinations
import matplotlib.pyplot as plt
import h5py
import os

def pad_z_dimension(image, desired_z_size, pad_value=0):
    current_z_size = image.shape[2]
    if current_z_size >= desired_z_size:
        return image[:, :, :desired_z_size]
    
    total_pad = desired_z_size - current_z_size
    pad_front = total_pad // 2
    pad_back = total_pad - pad_front
    
    padded_image = np.pad(image, ((0, 0), (0, 0), (pad_front, pad_back)), 'constant', constant_values=pad_value)
    
    return padded_image

def normalize_image(image):
    image = image - np.min(image)
    image = image / np.max(image)
    return image

# Parameters
num_modalities = 9  # Total number of channels (images + masks)
ff_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/ff/volumes/'
water_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/water/volumes/'
ff_r2_water_mask_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/ff/segmentations/'

ct_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/ct/volumes/'
ct_mask_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/ct/segmentations/'

mag_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/mag/volumes/'
mag_mask_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/mag/segmentations/'

pdff_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/pdff/volumes/'
pdff_mask_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/pdff/segmentations/'

t2_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/t2/volumes/'
t2_mask_folder = "C:/Users/Mittal/Desktop\multi-modal_data/t2/segmentations/"

axial_inphase_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/axial_inphase/volumes/'
axial_inphase_mask_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/axial_inphase/segmentations/'

axial_opposed_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/axial_opposed/volumes/'
axial_opposed_mask_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/axial_opposed/segmentations/'

portal_venous_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/portal_venous/volumes/'
portal_venous_mask_folder = 'C:/Users/Mittal/Desktop/multi-modal_data/portal_venous/segmentations/'

dummy_data_shape = (64, 64, 40)  # Shape for dummy data
dummy_data = np.zeros(dummy_data_shape)  # Dummy data array

def dice_coef(y_true, y_pred, smooth=1.):
  y_true_f = tf.keras.backend.flatten(y_true)
  y_pred_f = tf.keras.backend.flatten(y_pred)
  intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
# Define your folders for images and masks
image_folders = [ff_folder, ct_folder, mag_folder]
mask_folders = [ff_r2_water_mask_folder, ct_mask_folder, mag_mask_folder]

# Initialize lists to hold images and masks
num_modalities = 9  # 3 actual images + 6 dummy
dummy_data = np.zeros((64, 64, 40))

# Initialize lists for images and masks
whole_images = []
whole_masks = []

# Function to load images and masks
def load_images_and_masks(image_folder, mask_folder):
    images = []
    masks = []

    # Load images
    image_files = sorted(os.listdir(image_folder))
    for image_name in image_files:
        if image_name.endswith('.nii'):
            image_path = os.path.join(image_folder, image_name)
            image = nib.load(image_path)
            image_data = np.array(image.get_fdata())
            image_data = pad_z_dimension(image_data, 40, 0)  # Ensure proper z-dimension
            image_data = resize(image_data, (64, 64, 40))    # Resize to (64, 64, 40)
            image_data = normalize_image(image_data)        # Normalize
            images.append(np.array(image_data))

    # Load masks
    mask_files = sorted(os.listdir(mask_folder))
    for mask_name in mask_files:
        if mask_name.endswith('.nii'):
            mask_path = os.path.join(mask_folder, mask_name)
            mask = nib.load(mask_path)
            mask_data = np.array(mask.get_fdata())
            mask_data = pad_z_dimension(mask_data, 40, 0)
            mask_data = resize(mask_data, (64, 64, 40))
            mask_data = normalize_image(mask_data)
            masks.append(np.array(mask_data))

    return images, masks

# Load images and masks for each modality
all_images = []
all_masks = []

for i in range(len(image_folders)):
    images, masks = load_images_and_masks(image_folders[i], mask_folders[i])
    all_images.append(images)
    all_masks.append(masks)

# Create the whole_images and whole_masks with dummy data
for i in range(len(all_images[1])):  # Assuming all modalities have the same number of images
    whole_image = np.stack(( 
        dummy_data, all_images[1][i],
        dummy_data, dummy_data, dummy_data, dummy_data,
        dummy_data, dummy_data, dummy_data
    ), axis=-1)
    whole_images.append(whole_image)

for i in range(len(all_masks[1])):  # Assuming all modalities have the same number of masks
    whole_mask = np.stack(( 
        dummy_data, all_masks[1][i], 
        dummy_data, dummy_data, dummy_data, dummy_data, 
        dummy_data, dummy_data, dummy_data
    ), axis=-1)
    whole_masks.append(whole_mask)

# Print results before slicing
print("Number of whole images:", len(whole_images))
print("Number of whole masks:", len(whole_masks))

# Slicing the whole images and masks
sliced_image_dataset = []
sliced_mask_dataset = []

# Slicing along the depth dimension
for img in whole_images:
    for j in range(img.shape[2]):  # Iterate over depth slices
        sliced_image_dataset.append(img[:, :, j, :])

for mask in whole_masks:
    for j in range(mask.shape[2]):
        sliced_mask_dataset.append(mask[:, :, j, :])

# Convert to numpy arrays
sliced_image_dataset = np.array(sliced_image_dataset)[..., np.newaxis]
sliced_mask_dataset = np.array(sliced_mask_dataset)[..., np.newaxis]

# Print the number of slices created
print("Number of sliced images:", len(sliced_image_dataset))
print("Number of sliced masks:", len(sliced_mask_dataset))

# Load your pre-trained model from the HDF5 file
model = tf.keras.models.load_model("D:/Downloads/multimodal_1.h5",  custom_objects={'dice_coef': dice_coef})
# Make predictions using images
predictions = model.predict(sliced_image_dataset)

dice_scores = []
for i in range(len(predictions)):
    predicted_mask = (predictions[i] > 0.5).astype(np.float32)   # Thresholding
    true_mask = sliced_mask_dataset[i].squeeze().astype(np.float32)  # Adjust dimensions as necessary
    
    # Calculate the Dice coefficient
    score = dice_coef(true_mask, predicted_mask)
    dice_scores.append(score)

# Print average Dice score
print("Average Dice Coefficient:", np.mean(dice_scores))

sample_index = 123

# Get the input image, ground truth mask, and prediction for this sample
input_image = sliced_image_dataset[sample_index]
ground_truth_mask = sliced_mask_dataset[sample_index]
prediction = predictions[sample_index]

# Function to plot an image
def plot_image(ax, img, title):
    #if img.shape[-1] == 9:  # If it's a 9-channel image
    img = img[:,:,1]  # Take only the first channel
    im = ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    return im

# Create a figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

# Plot input image
im1 = plot_image(ax1, input_image, 'Input Image (First Channel)')
fig.colorbar(im1, ax=ax1)

# Plot ground truth mask
im2 = plot_image(ax2, ground_truth_mask, 'Ground Truth Mask (First Channel)')
fig.colorbar(im2, ax=ax2)

# Plot prediction
im3 = plot_image(ax3, prediction, 'Model Prediction (First Channel)')
fig.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.show()