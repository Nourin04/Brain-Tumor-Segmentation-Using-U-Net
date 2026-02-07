
# Brain Tumor Segmentation Using U-Net 


This repository contains a **PyTorch implementation of a U-Net model for brain tumor segmentation**, developed and executed entirely within a Jupyter notebook.
The project performs **binary semantic segmentation** on brain tumor images using polygon annotations converted from **YOLO-format labels** into pixel-wise masks.

---

## Objective

To train a **U-Net convolutional neural network** that predicts a **binary segmentation mask** identifying tumor regions from input brain images.

---

## Dataset Source and Annotation

* The raw dataset was **downloaded from Kaggle**
* Images were **manually annotated using Roboflow**
* Annotations were exported in **YOLO polygon format**
* The annotated dataset was then **loaded locally into the notebook** for preprocessing and training

---

## Dataset Structure and Annotation Format

The dataset used in the notebook follows the structure below:

```
Brain-tumor-segmentation-2/
├── train/
│   ├── images/
│   ├── labels/   (YOLO polygon annotations)
│   └── masks/    (generated during preprocessing)
├── test/
│   ├── images/
│   ├── labels/
│   └── masks/    (generated during preprocessing)
```

### Mask Generation Process

* YOLO polygon coordinates are:

  * Normalized
  * Converted into pixel coordinates
* Polygons are filled using `cv2.fillPoly`
* Generated masks are saved as grayscale images
* Pixel values:

  * Tumor region: **255**
  * Background: **0**

---

## Data Visualization

The notebook includes visualization of:

* Randomly selected training images
* Corresponding generated segmentation masks

This step is used to verify correct alignment between images and masks.

---

## Dataset and DataLoader Implementation

### Custom PyTorch Dataset

A custom `BrainTumorDataset` class is implemented using `torch.utils.data.Dataset`, which:

* Loads images using OpenCV
* Converts images from BGR to RGB
* Loads corresponding masks
* Converts masks to binary format
* Expands mask dimensions for model compatibility

---

### Data Preprocessing and Augmentation

**Training Transformations (Albumentations):**

* Resize to **256 × 256**
* Horizontal flip (p = 0.5)
* Random brightness/contrast (p = 0.2)
* Normalization
* Conversion to PyTorch tensors

**Testing Transformations:**

* Resize to **256 × 256**
* Normalization
* Conversion to PyTorch tensors

---

### DataLoader Configuration

* Training batch size: **4**
* Training shuffle: Enabled
* Testing batch size: **1**
* Workers configured through PyTorch `DataLoader`

---

## U-Net Model Architecture

The notebook implements a **U-Net architecture from scratch using PyTorch**.

### Architecture Components

* **DoubleConv**
  Two convolution layers with:

  * Batch Normalization
  * ReLU activation

* **Down Block**
  MaxPooling followed by DoubleConv

* **Up Block**
  Bilinear upsampling
  Feature map concatenation with encoder outputs
  Followed by DoubleConv

* **Output Layer**
  1×1 convolution producing a single-channel output

---

### Model Configuration

* Input channels: **3**
* Output channels: **1**
* Base number of filters: **64**
* Upsampling method: **Bilinear interpolation**
* Output type: **Logits (sigmoid applied during loss computation)**


---

## Loss Function and Optimizer

* **Loss Function:**
  `BCEWithLogitsLoss`

* **Optimizer:**
  Adam optimizer
  Learning rate = **1e-4**

---

## Evaluation Metric

### Dice Coefficient

* Predictions are passed through a sigmoid function
* Threshold applied at **0.5**
* Dice coefficient computed for each batch
* Average Dice score reported during training

---

## Training Configuration

* Number of epochs: **25**
* Training loop includes:

  * Forward pass
  * Loss computation
  * Backpropagation
  * Parameter updates
  * Dice score calculation
* Training progress is logged per epoch

---


## Inference on Test Data

For test samples:

1. YOLO polygon annotations are converted into masks
2. Test transformations are applied
3. Model predictions are generated
4. Sigmoid activation and thresholding are applied
5. Results are visualized

### Output Visualization

For each test image:

* Original image
* Ground truth mask
* Predicted segmentation mask

---

## Libraries and Tools Used

* Python
* PyTorch
* Torchvision
* Albumentations
* OpenCV
* NumPy
* Matplotlib
* Roboflow
* Kaggle
* tqdm




