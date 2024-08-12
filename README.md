
# Detection of Dental Abnormalities and Teeth Segmentation Using Deep Learning

## Overview

This project explores the application of deep learning techniques to detect dental abnormalities and segment teeth from radiographic images. We focus on a binary classification problem to determine if a radiograph contains a disease or other abnormalities, as well as on segmenting the teeth in these images. The images used in this project were labeled by experts from the School of Dental Medicine - Tufts University, identifying conditions such as caries, periapical infections, pathological migration, altered bone height, and bony impactions of third molars.
## Key Features

* Teeth Segmentation:

    * Utilizes ResNet-50 as a feature extractor with DeepLabV3 for generating segmentation masks, achieving 95% accuracy, 73% F1-score, and 92% IOU.

    * Comparison with U-Net showed a dice mean of 90.7% for segmentation tasks.

* Abnormality Detection:

    * Achieved 87% accuracy in detecting dental abnormalities using a U-Net model.
## Motivation
The motivation for this project lies in the potential of deep learning to enable early detection of subtle changes in dental images, leading to improved patient outcomes. By automating the analysis of large volumes of data, dental professionals can focus more on treatment planning and personalized care.
## Database
We utilized the Tufts Dental Dataset (TDD), released in April 2022, which includes:

    * 1000 panoramic X-rays labeled with bounding boxes for the maxillomandibular region, teeth, and specific abnormalities.
    * Although the dataset includes gaze plots from eye-tracking data and textual descriptions of abnormalities, our project focuses solely on the X-ray images and their labels.
## Installation

1. Clone the repository:

```bash
https://github.com/shri11081999/Detection-of-Dental-Abnormalities-and-Teeth-segmentaion.git
```
2. Install the required dependencies:

* Python 3.7 or later
* TensorFlow, PyTorch, Keras
* OpenCV or scikit-image for image preprocessing
* Visual Studio as the IDE


3. Download the Dataset

Acquire the TDD dataset and place it in the data/ directory.


## Contributing

Feel free to make any changes in the project.
## Demo

![gui](https://github.com/user-attachments/assets/a66109bf-e14d-4109-a005-0b04cc890c17)


## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Results

* **Teeth Segmentation**: Achieved state-of-the-art results with ResNet-50 and DeepLabV3, attaining 95% accuracy, 73% F1-score, and 92% IOU.
* **Abnormality Detection**: The U-Net model achieved an accuracy of 87% in detecting dental abnormalities.

## Contact

For any questions or issues, please contact dixitshriniket976@gmail.com.

