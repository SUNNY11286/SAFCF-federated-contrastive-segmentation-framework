# 3D Medical Image Segmentation

It is federated framework for the segmentation task, which uses novel loss and aggregation strategies.



This project implements a 3D U-Net model for medical image segmentation using TensorFlow and the `segmentation_models_3D` library. The project is structured into multiple modules for better maintainability and readability.

## Project Structure

- `data_loader.py`: Contains the `DataLoader` class for loading and preprocessing the dataset.
- `model.py`: Contains the `UNET3D` class for defining the 3D U-Net model.
- `train.py`: Contains the `Trainer` class for training and evaluating the model.
- `utils.py`: Contains utility functions for metrics and loss calculations.
- `main.py`: The main script to run the training process.

## Requirements

- Python 3.6+
- TensorFlow 2.x
- segmentation_models_3D
- numpy
- tqdm

You can install the required packages using the following command:

```pip
pip install -r requirements.txt
```



## Usage

1. **Clone the repository:**

   ```git
   https://github.com/SUNNY11286/SAFCF-federated-contrastive-segmentation-framework.git
   ```
2. **Prepare your dataset:**

Ensure your dataset is structured as follows:

```
/kaggle/input/
├── msd-reshaped-64/
│   ├── task04_hippocampus_64/
│   │   ├── images/
│   │   └── masks/
│   ├── task02_heart_64/
│   │   ├── images/
│   │   └── masks/
├── msd01-64/
│   ├── processed_64/
│   │   ├── images/
│   │   └── masks/
```

3. **Run the training script:**
   ```
   python main.py
   ```
   

## Customization

- **DataLoader**: Modify `data_loader.py` to change how data is loaded and preprocessed.
- **Model**: Modify `model.py` to change the architecture of the 3D U-Net model.
- **Training**: Modify `train.py` to change the training process, including the loss function, metrics, and training loop.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.



