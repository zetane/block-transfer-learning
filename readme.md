# Image Classification with Pre-trained ResNet50

This project demonstrates how to use a pre-trained ResNet50 model for image classification. It includes downloading the model, loading image data, performing predictions, and saving the results with predicted labels.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- PIL (Pillow)
- requests

## Setup

1. Clone the repository:

    ```sh
    git clone https://github.com/your-repo/image-classification-resnet50.git
    cd image-classification-resnet50
    ```

2. Install the required packages:

    ```sh
    pip install torch torchvision pillow requests
    ```

3. Ensure you have a folder named `test_set` in the root directory containing the images you want to classify. The folder structure should follow the format expected by `ImageFolder`, i.e., images organized in subfolders per class (though the class labels won't be used in this example).

## Usage

### Downloading the Pre-trained Model

The script checks if the ResNet50 model pre-trained on ImageNet is present. If not, it downloads the model.

### Transforming the Data

Images are resized to 224x224 pixels, converted to tensors, and normalized using the mean and standard deviation of the ImageNet dataset.

### Loading the Dataset

The `test_set` folder is loaded using `torchvision.datasets.ImageFolder`, and a `DataLoader` is created for batch processing.

### Evaluating the Model

The `compute` function evaluates the model on the test images, adds the predicted labels to the images, and saves the results in the `output` folder.



