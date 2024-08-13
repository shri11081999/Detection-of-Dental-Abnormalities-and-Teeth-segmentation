# 🦷 Detection of Dental Abnormalities and Teeth Segmentation Using Deep Learning

## 📄 Overview

This project explores the application of deep learning techniques to detect dental abnormalities and segment teeth from radiographic images. The focus is on:

- **Binary Classification:** Determine if a radiograph contains a disease or other abnormalities.
- **Teeth Segmentation:** Segment the teeth in radiographic images.

The images used were expertly labeled by the **School of Dental Medicine - Tufts University**, identifying conditions such as caries, periapical infections, pathological migration, altered bone height, and bony impactions of third molars.

## ✨ Key Features

### 🦷 Teeth Segmentation:
- **Model:** Utilizes ResNet-50 as a feature extractor with DeepLabV3 for generating segmentation masks.
- **Performance:** 
  - 95% Accuracy
  - 73% F1-score
  - 92% IOU
- **Comparison:** U-Net showed a dice mean of 90.7% for segmentation tasks.

### 🔍 Abnormality Detection:
- **Model:** Achieved 87% accuracy in detecting dental abnormalities using a U-Net model.

## 🎯 Motivation

The motivation for this project lies in the potential of deep learning to enable early detection of subtle changes in dental images, leading to improved patient outcomes. By automating the analysis of large volumes of data, dental professionals can focus more on treatment planning and personalized care.

## 🗂️ Database

We utilized the **Tufts Dental Dataset (TDD)**, released in April 2022, which includes:

- **1000 panoramic X-rays** labeled with bounding boxes for the maxillomandibular region, teeth, and specific abnormalities.
- While the dataset includes gaze plots from eye-tracking data and textual descriptions of abnormalities, our project focuses solely on the X-ray images and their labels.

🔒 **Note:** The dataset is proprietary and cannot be shared publicly due to industry confidentiality.

## 🛠️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/shri11081999/Detection-of-Dental-Abnormalities-and-Teeth-segmentaion.git

2. **Install the required dependencies:**

   - Python 3.7 or later
   - TensorFlow, PyTorch, Keras
   - OpenCV or scikit-image for image preprocessing
   - Visual Studio as the IDE
   - 
3. **Download the Dataset:**

   Acquire the TDD dataset and place it in the `teeth database/` directory.

## 🤝 Contributing

Feel free to contribute to this project by making any changes, improvements, or suggestions.

## 🎥 Demo

![GUI Demo](https://github.com/user-attachments/assets/a66109bf-e14d-4109-a005-0b04cc890c17)

## 📸 Screenshots

![Teeth Segmentation 1](https://github.com/user-attachments/assets/e11578b3-b9be-4467-b42a-860fe4dbcd31)
![Teeth Segmentation 3](https://github.com/user-attachments/assets/3fde82c2-8456-4c97-8259-4efb076fe10f)

## 📊 Results

- **Teeth Segmentation:** 
  - **Model:** ResNet-50 + DeepLabV3
  - **Results:** 95% Accuracy, 73% F1-score, 92% IOU

- **Abnormality Detection:** 
  - **Model:** U-Net
  - **Results:** 87% Accuracy

  ## 📧 Contact

For any questions or issues, please reach out to 📬 [dixitshriniket976@gmail.com](mailto:dixitshriniket976@gmail.com).

