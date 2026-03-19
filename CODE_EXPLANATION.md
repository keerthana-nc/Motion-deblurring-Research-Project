# LOL-Blur Dataloader Code Explanation

## Overview
This code creates a PyTorch Dataset class (`LOLBlurDataset`) that loads image pairs for motion deblurring tasks. It pairs blurred/noisy images with their corresponding sharp ground truth images.

## Step-by-Step Breakdown

### 1. **Class Initialization (`__init__`)**
   - **Purpose**: Sets up the dataset by finding and organizing image pairs
   - **Inputs**: 
     - `root`: Path to the dataset root directory
     - `image_size`: Target size for resizing images (default: 256x256)
   
   **Steps:**
   - **Step 1.1**: Constructs paths to two directories:
     - `low_blur_noise/` - Contains blurred/noisy input images
     - `high_sharp_original/` - Contains sharp ground truth images
   
   - **Step 1.2**: Validates that both directories exist (raises error if missing)
   
   - **Step 1.3**: Recursively searches for image pairs:
     - Iterates through each subfolder in `low_blur_noise/`
     - For each subfolder, looks for image files (PNG, JPG, JPEG)
     - Checks if a matching file exists in the corresponding `high_sharp_original/` subfolder
     - Only adds pairs where both images exist
     - Skips files without matches (with warning)
   
   - **Step 1.4**: Validates that at least one image pair was found
   
   - **Step 1.5**: Defines image transformations:
     - Resize to `image_size x image_size`
     - Convert PIL Image to PyTorch Tensor
     - Normalize pixel values from [0,1] to [-1,1] range (using mean=0.5, std=0.5)
   
   - **Step 1.6**: Prints dataset statistics (root path, number of pairs, image size)

### 2. **`__len__` Method**
   - Returns the total number of image pairs in the dataset
   - Required by PyTorch Dataset interface

### 3. **`__getitem__` Method**
   - **Purpose**: Retrieves a single image pair at a given index
   - **Input**: `idx` - Index of the image pair to retrieve
   - **Process**:
     - Opens the blurred input image and converts to RGB
     - Opens the corresponding ground truth image and converts to RGB
     - Applies the same transformations to both images
     - Returns a dictionary containing:
       - `"input"`: Transformed blurred image tensor
       - `"gt"`: Transformed ground truth image tensor
       - `"filename"`: Name of the input image file

### 4. **Main Block (Testing)**
   - Creates a dataset instance
   - Wraps it in a DataLoader (batch_size=4, shuffles data)
   - Tests by loading 2 batches and printing their shapes and filenames

## Data Flow

```
Dataset Root
├── low_blur_noise/          (Input: Blurred images)
│   ├── subfolder1/
│   │   ├── img1.png
│   │   └── img2.png
│   └── subfolder2/
│       └── img3.png
└── high_sharp_original/      (Ground Truth: Sharp images)
    ├── subfolder1/
    │   ├── img1.png          (matches low_blur_noise/subfolder1/img1.png)
    │   └── img2.png          (matches low_blur_noise/subfolder1/img2.png)
    └── subfolder2/
        └── img3.png          (matches low_blur_noise/subfolder2/img3.png)
```

## Use Case
This dataset is designed for training deep learning models that learn to:
- **Input**: Blurred/noisy image
- **Output**: Sharp, deblurred image
- The model learns the mapping from blurred → sharp by comparing predictions to ground truth during training
