import os
import tensorflow as tf
import segmentation_models_3D as sm
from data_loader import DataLoader
from model import UNET3D
from train import Trainer
from utils import dice_coef, precision, sensitivity, specificity

def main():
    base_path = "/kaggle/input"
    datasets = ["msd-reshaped-64/task04_hippocampus_64", "msd-reshaped-64/task02_heart_64", "msd01-64/processed_64"]
    data_loader = DataLoader(base_path)

    clients = []
    for dataset in datasets:
        images, masks = data_loader.load_data(dataset)
        clients.append({
            "name": dataset,
            "model": UNET3D(1).call(tf.keras.layers.Input(shape=(64, 64, 64, 1))),
            "optimizer": tf.keras.optimizers.Adam(1e-5),
            "original_data": data_loader.load_img(images[:10]),  # Adjust batch size as needed
            "length_ratio": len(images) / sum(len(data_loader.load_data(d)[0]) for d in datasets)
        })

    server = {
        "model": UNET3D(1).call(tf.keras.layers.Input(shape=(64, 64, 64, 1))),
        "optimizer": tf.keras.optimizers.Adam(1e-5)
    }

    test_data_gen = data_loader.load_img(clients[0]["original_data"][:10])  # Adjust test data as needed

    trainer = Trainer(
        model=UNET3D(1),
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss_fn=sm.losses.DiceLoss() + sm.losses.BinaryFocalLoss(),
        metrics=[dice_coef, precision, sensitivity, specificity],
        batch_size=5,
        epochs=10,
        num_clients=3,
        num_comm=3
    )

    trainer.train(clients, server, test_data_gen)

if __name__ == "__main__":
    main()