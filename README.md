# ml4h_2

### Part_1

The code of every question is fully contained in a notebook named 1_Qi.ipynb.

## Pneumonia Classification with Explainability (ML4H Project 2 - Task 2)

###  Overview

This project explores interpretable and explainable deep learning techniques for medical imaging using the **Chest X-Ray Images (Pneumonia)** dataset. The objective is to train convolutional neural networks (CNNs) for pneumonia classification and analyze their decision-making processes using post-hoc explainability methods such as **Integrated Gradients** and **Grad-CAM**.

---

### 📁 Directory Structure

task_2/
├── data/
│ ├── train/
│ │ ├── NORMAL/
│ │ └── PNEUMONIA/
│ ├── test/
│ │ ├── NORMAL/
│ │ └── PNEUMONIA/
│ └── val/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── pneumonia_cnn_model_15epochs.pth
├── pneumonia_CNN_model_random_label_epochs_15.pth
├── Q1.ipynb
├── Q2_CNN.ipynb
├── Q3_CNN.ipynb
├── Q4_CNN.ipynb
├── Q5_CNN.ipynb


---

## 📝 Notebooks Description

| Notebook       | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `Q1.ipynb`     | Exploratory Data Analysis (EDA): Data inspection, visualization, and preprocessing. |
| `Q2_CNN.ipynb` | CNN training and evaluation for binary pneumonia classification.             |
| `Q3_CNN.ipynb` | Integrated Gradients implementation with attribution map visualization.      |
| `Q4_CNN.ipynb` | Grad-CAM implementation and comparison with Integrated Gradients.            |
| `Q5_CNN.ipynb` | Data Randomization Test to validate attribution method reliability.          |

### Download and Prepare Data
Download the dataset from Kaggle and place it under the data/ directory as shown above.

## Run the Notebooks
Execute the notebooks in the following order for a full analysis:
Q1.ipynb → Q2_CNN.ipynb → Q3_CNN.ipynb → Q4_CNN.ipynb → Q5_CNN.ipynb

### Features

Convolutional Neural Network (CNN) for binary classification.  
- Post-hoc Explainability Methods:  
  - Integrated Gradients (Q3)  
  - Grad-CAM (Q4)    
Sanity Checks with Data Randomization Test (Q5)  
Clean, reproducible pipeline with model saving and attribution comparison.  