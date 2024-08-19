import os
import numpy as np

class DataLoader:
    def __init__(self, base_path):
        self.base_path = base_path

    def load_data(self, dataset_name):
        images_path = os.path.join(self.base_path, dataset_name, "images")
        masks_path = os.path.join(self.base_path, dataset_name, "masks")
        
        images_files = os.listdir(images_path)
        masks_files = os.listdir(masks_path)
        
        npy_images_path = [os.path.join(images_path, img_file) for img_file in images_files]
        npy_masks_path = [os.path.join(masks_path, mask_file) for mask_file in masks_files]
        
        npy_images_path.sort()
        npy_masks_path.sort()
        
        return npy_images_path, npy_masks_path

    def load_img(self, img_list):
        images = []
        for image_name in img_list:
            image = np.load(image_name)
            if image.ndim == 3:
                image = np.expand_dims(image, axis=-1)
            target_shape = (64, 64, 64, 1)
            image = np.pad(image, ((0, max(0, target_shape[0] - image.shape[0])),
                                   (0, max(0, target_shape[1] - image.shape[1])),
                                   (0, max(0, target_shape[2] - image.shape[2])),
                                   (0, 0)), mode='constant')
            image = image[:target_shape[0], :target_shape[1], :target_shape[2], :]
            images.append(image)
        if len(set(img.shape for img in images)) != 1:
            raise ValueError("All images must have the same shape.")
        return np.asarray(images, dtype=np.float32)