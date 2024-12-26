# Image Classification for Cattle CVD Detection via Retina Images

## Project Overview
This project involves creating a deep learning model to classify cattle retina images into two categories: `healthy` and `diseased`, to detect cardiovascular diseases (CVD) in cattle. The solution includes dataset preparation, preprocessing, model training, evaluation, and Grad-CAM visualization for model interpretability.

## Objectives
1. **Dataset Identification**: Generate synthetic retina images for healthy and diseased classes due to the unavailability of a real dataset.
2. **Image Preprocessing**: Apply resizing, normalization, and data augmentation to improve model performance.
3. **Model Development**: Build a Convolutional Neural Network (CNN) using transfer learning (VGG16) for classification.
4. **Evaluation**: Evaluate the model using metrics such as accuracy, precision, recall, and AUC-ROC.
5. **Visualization**: Use Grad-CAM to visualize regions of interest influencing model predictions.

## Dataset
### **Synthetic Dataset Creation**
- The dataset was generated using the `create_synthetic_retina_image` function to simulate retina patterns for two classes:
  - `healthy`
  - `diseased`
- Features include:
  - Random circular patterns to mimic retinal structures.
  - Random lines to simulate blood vessels.
- **Dataset Structure**:
  - **Training Data**: 100 images per class.
  - **Testing Data**: 20 images per class.

### **Challenges**
- Lack of real-world data.
- Synthetic data may not fully capture the complexity of real retina patterns.

## Preprocessing
- Resizing images to `(224, 224)` to match the input size of the VGG16 model.
- Normalizing pixel values to the range `[0, 1]`.
- Data augmentation:
  - Rotation, shear, zoom, width/height shift, and horizontal flipping.

## Model Architecture
- **Base Model**: Pre-trained VGG16 (transfer learning, `include_top=False`).
- **Custom Layers**:
  - Flatten layer.
  - Dense layer with 128 units and ReLU activation.
  - Dropout layer (0.5).
  - Output layer with 1 unit and sigmoid activation for binary classification.
- **Compilation**:
  - Optimizer: Adam.
  - Loss Function: Binary Crossentropy.
  - Metrics: Accuracy.

## Training
- **Early Stopping**:
  - Monitor: Validation loss.
  - Patience: 5 epochs.
  - Restores the best weights.
- **Epochs**: 20.

## Evaluation
- **Metrics**:
  - Accuracy: 53%.
  - AUC-ROC: 0.53.
  - Precision, recall, and F1-score.
- **Insights**:
  - Imbalanced precision and recall suggest data quality issues or the need for further tuning.

## Grad-CAM Visualization
- Generated Grad-CAM heatmaps to identify regions of interest influencing predictions.
- Visualizations showed model focus areas on the synthetic retina images.

## Tools and Libraries
- Python
- TensorFlow/Keras
- NumPy
- OpenCV
- Matplotlib
- TQDM

## Directory Structure
```
project/
├── synthetic_cattle_retina_dataset/
│   ├── train/
│   │   ├── healthy/
│   │   └── diseased/
│   └── test/
│       ├── healthy/
│       └── diseased/
└── scripts/
    ├── data_preprocessing.py
    └── model_training.py
```

## How to Run the Project
1. Clone the repository.
   ```bash
   git clone <repository_url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dataset generation script:
   ```bash
   python data_preprocessing.py
   ```
4. Train the model:
   ```bash
   python model_training.py
   ```
5. Evaluate the model and visualize Grad-CAM heatmaps.

## Observations
- The model's performance is limited due to synthetic data quality and size.
- Grad-CAM visualizations provide insights into the model's decision-making.

## Future Improvements
- Use real-world retina datasets for better performance.
- Enhance synthetic data generation with more realistic patterns.
- Experiment with different architectures and hyperparameters.

## Conclusion
This project demonstrates the use of synthetic data and transfer learning to detect CVD in cattle using retina images. While the results are preliminary, it provides a foundation for further work with real-world datasets.

