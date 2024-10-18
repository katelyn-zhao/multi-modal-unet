import os
import io
import random
import nibabel as nib
import numpy as np
from skimage.transform import resize
from itertools import combinations
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from sklearn.model_selection import KFold
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

def dice_coef(y_true, y_pred, smooth=1.):
  y_true_f = tf.keras.backend.flatten(y_true)
  y_pred_f = tf.keras.backend.flatten(y_pred)
  intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def tpr(y_true, y_pred, threshold=0.5):
    y_pred_pos = tf.cast(y_pred > threshold, tf.float32)
    y_true_pos = tf.cast(y_true > threshold, tf.float32)
    
    true_pos = tf.reduce_sum(tf.cast(tf.logical_and(y_true_pos == 1, y_pred_pos == 1), tf.float32))
    actual_pos = tf.reduce_sum(tf.cast(y_true_pos, tf.float32))
    
    tpr = true_pos / (actual_pos + tf.keras.backend.epsilon())
    return tpr

def fpr(y_true, y_pred, threshold=0.5):
    y_pred_pos = tf.cast(y_pred > threshold, tf.float32)
    y_true_neg = tf.cast(y_true <= threshold, tf.float32)
    
    false_pos = tf.reduce_sum(tf.cast(tf.logical_and(y_true_neg == 1, y_pred_pos == 1), tf.float32))
    actual_neg = tf.reduce_sum(tf.cast(y_true_neg, tf.float32))
    
    fpr = false_pos / (actual_neg + tf.keras.backend.epsilon())
    return fpr

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


all_image_folders = [ff_folder, ct_folder, pdff_folder, water_folder, mag_folder, t2_folder, axial_inphase_folder, axial_opposed_folder, portal_venous_folder]
all_mask_folders = [ff_r2_water_mask_folder, ct_mask_folder, pdff_mask_folder, ff_r2_water_mask_folder, mag_mask_folder, t2_mask_folder, axial_inphase_mask_folder, axial_opposed_mask_folder, portal_venous_mask_folder]


ff_images = []
water_images = []
ct_images = []
mag_images = []
pdff_images = []
t2_images = []
axial_inphase_images = []
axial_opposed_images = []
portal_venous_images = []

all_images = [ff_images, ct_images, pdff_images, water_images, mag_images, t2_images, axial_inphase_images,axial_opposed_images, portal_venous_images]

ff_masks = []
water_masks = []
ct_masks = []
mag_masks = []
pdff_masks = []
t2_masks = []
axial_inphase_masks = []
axial_opposed_masks = []
portal_venous_masks = []

all_masks = [ff_masks, ct_masks, pdff_masks, water_masks, mag_masks, t2_masks, axial_inphase_masks, axial_opposed_masks, portal_venous_masks]

for i in range(len(all_image_folders)):
    image = sorted(os.listdir(all_image_folders[i]))
    for j, image_name in enumerate(image):
        if (image_name.split('.')[1] == 'nii' and j !=4):
            image = nib.load(all_image_folders[i]+image_name)
            image = np.array(image.get_fdata())
            image = pad_z_dimension(image, 40, 0)
            image = resize(image, (64, 64, 40))
            image = normalize_image(image)
            all_images[i].append(np.array(image))

for i in range(len(all_mask_folders)):
    mask = sorted(os.listdir(all_mask_folders[i]))
    for j, image_name in enumerate(mask):
        if (image_name.split('.')[1] == 'nii' and j != 4):
            image = nib.load(all_mask_folders[i]+image_name)
            image = np.array(image.get_fdata())
            image = pad_z_dimension(image, 40, 0)
            image = resize(image, (64, 64, 40))
            image = normalize_image(image)
            all_masks[i].append(np.array(image))

whole_images = []
whole_masks = []

print(len(water_images))
print(len(water_masks))


for i in range(len(water_images)):
    whole_image = np.stack((ff_images[i], ct_images[i], pdff_images[i],
    water_images[i], mag_images[i], t2_images[i], axial_inphase_images[i], 
    axial_opposed_images[i], portal_venous_images[i]), axis=-1)
    whole_images.append(whole_image)

for i in range(len(water_images)):
    whole_mask = np.stack((ff_masks[i], ct_masks[i], pdff_masks[i],
    water_masks[i], mag_masks[i], t2_masks[i], axial_inphase_masks[i], 
    axial_opposed_masks[i], portal_venous_masks[i]), axis=-1)
    whole_masks.append(whole_mask)

del all_masks, all_images, ff_images, ct_images, pdff_images, water_images, mag_images, t2_images
del axial_inphase_images, axial_opposed_images, portal_venous_images, ff_masks, ct_masks, pdff_masks, water_masks, mag_masks
del t2_masks, axial_inphase_masks, axial_opposed_masks, portal_venous_masks

print(len(whole_images))
print(len(whole_masks))

def generate_all_masks(num_modalities=9):
    all_masks = []

    for n in range(1, num_modalities + 1):
        for indices in combinations(range(num_modalities), n):
            mask = np.zeros(num_modalities, dtype=int)
            mask[list(indices)] = 1 
            all_masks.append(mask)

    return all_masks


masks = generate_all_masks()

combination_images = []
combination_masks = []

for i in range(len(whole_images)):
    for mask in masks:
        single_combination_image = whole_images[i] * mask[np.newaxis, np.newaxis, np.newaxis, :]
        single_combination_mask = whole_masks[i] * mask[np.newaxis, np.newaxis, np.newaxis, :]

        combination_images.append(single_combination_image)
        combination_masks.append(single_combination_mask)

del whole_images, whole_masks

sliced_image_dataset = []
sliced_mask_dataset = []

combination_images = np.array(combination_images)
combination_masks = np.array(combination_masks)

for i in range(len(combination_images)):
    for j in range(combination_images[i].shape[2]):
        sliced_image_dataset.append(combination_images[i][:,:,j,:])
        sliced_mask_dataset.append(combination_masks[i][:,:,j,:])

del combination_images, combination_masks

sliced_image_dataset = np.array(sliced_image_dataset)[..., np.newaxis]
sliced_mask_dataset = np.array(sliced_mask_dataset)[..., np.newaxis]

print(len(sliced_image_dataset))
print(len(sliced_mask_dataset))

def simple_unet_model(IMG_HEIGHT=64, IMG_WIDTH=64, IMG_CHANNELS=9):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    #Contraction path
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(IMG_CHANNELS, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    
    model.summary()
    
    return model

IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 9

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

n_splits = 5

kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

for i, (train_index, test_index) in enumerate(kf.split(sliced_image_dataset, sliced_mask_dataset)):
    X_train, X_test = sliced_image_dataset[train_index], sliced_image_dataset[test_index]
    y_train, y_test = sliced_mask_dataset[train_index], sliced_mask_dataset[test_index]

    f = open(f"C:/Users/Mittal/Desktop/kunet/multimodal_output.txt", "a")
    print("FOLD----------------------------------", file=f)
    print("x-training: ", len(X_train), file=f)
    print("x-testing: ", len(X_test), file=f)
    print("y-training: ", len(y_train), file=f)
    print("y-testing: ", len(y_test), file=f)
    f.close()

    model = get_model()

    checkpoint = ModelCheckpoint(f'C:/Users/Mittal/Desktop/kunet/multimodal_{i}.h5', monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        batch_size=128,
                        verbose=1,
                        epochs=300,
                        validation_data=(X_test, y_test),
                        shuffle=False,
                        callbacks=[checkpoint, early_stopping])

    model_save_path = f'C:/Users/Mittal/Desktop/kunet/multimodal_final_{i}.h5'

    model.save(model_save_path)
    print(f'Model for fold {i} saved to {model_save_path}')

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], color='r')
    plt.plot(history.history['val_loss'])
    plt.ylabel('Losses')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val.'], loc='upper right')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coef'], color='r')
    plt.plot(history.history['val_dice_coef'])
    plt.ylabel('dice_coef')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(f'C:/Users/Mittal/Desktop/kunet/multimodal_process{i}.png')
    plt.close()

    max_dice_coef = max(history.history['dice_coef'])
    max_val_dice_coef = max(history.history['val_dice_coef'])

    f = open(f"C:/Users/Mittal/Desktop/kunet/multimodal_output.txt", "a")
    print("max dice coef: ", max_dice_coef, file=f)
    print("max val dice coef: ", max_val_dice_coef, file=f)
    f.close()

    del X_train, X_test, y_train, y_test