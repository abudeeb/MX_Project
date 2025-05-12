# Intrusion Detection System (IDS) with xNIDS Integration

## Overview
This project integrates a Convolutional Neural Network (CNN)–based Intrusion Detection System (IDS) model with **xNIDS**, a deep-learning explanation framework.  
Users can:

- Train an IDS on the Usenix Modbus dataset  
- Use the trained model to predict malicious activity  
- Generate and visualize explanations for the model’s predictions  

---

## Table of Contents
1. [Part 1: IDS Model (CNN)](#part-1-ids-model-cnn)  
2. [Part 2: xNIDS Integration](#part-2-xnids-integration)  
3. [System Requirements](#system-requirements)  
4. [Installation](#installation)  
5. [Setup & Execution](#setup--execution)  
   - [1. Prepare the Dataset](#1-prepare-the-dataset)  
   - [2. Train the Model](#2-train-the-model)  
   - [3. Generate Explanations](#3-generate-explanations)  
   - [4. Visualize Results](#4-visualize-results)  
6. [Troubleshooting](#troubleshooting)  
7. [License](#license)  

---

## Part 1: IDS Model (CNN)
1. **Dataset**  
   - Trained on the Usenix Modbus dataset.  
2. **Model**  
   - Convolutional Neural Network for multi-class classification.  
3. **Labels**  
   - Send fake command  
   - Moving Two files  
   - Exploit ms08 netapi  
   - CnC uploading exe  
   - Characterization  
   - Least Significant bit exfiltration  

---

## Part 2: xNIDS Integration
1. **xNIDS**  
   - Open-source explainability framework for deep-learning–based NIDS.  
2. **Explanation**  
   - Highlights the top features influencing each prediction.  
3. **Visualization**  
   - Generates bar charts of feature importances.  

---

## System Requirements
- **Python** 3.9 or higher  
- **Dependencies**  
  - tensorflow 2.x  
  - numpy  
  - pandas  
  - matplotlib  
  - scikit-learn  
  - absl-py  
  - ASGL (used by `explanation.py`)  

---

## Installation
1. **Clone this repository**  
   ```bash
   git clone https://github.com/antoine-lemay/Modbus_dataset.git
   cd Modbus_dataset

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Install Jupyter Notebook** (if needed)

   ```bash
   pip install jupyter
   ```

---

## Setup & Execution

### 1. Prepare the Dataset

* Follow instructions in `IDS.ipynb` to preprocess:

  * Fill missing values
  * One-hot encode categorical features (e.g., `query_type`, `Protocol`)
  * Normalize with `StandardScaler`

### 2. Train the Model

1. Open `IDS.ipynb`
2. Train the CNN on the preprocessed data
3. Save the model as `IDS1_model.h5`
4. Evaluate performance (accuracy, precision, etc.)

### 3. Generate Explanations

1. Clone xNIDS:

   ```bash
   git clone https://github.com/CactiLab/code-xNIDS
   cd code-xNIDS
   pip install -r requirements.txt
   ```
2. Copy `IDS1_model.h5` into the `code-xNIDS` folder.
3. Open `XNIDS.ipynb`, load your model and test data (`X_test_cnn.csv`, `y_test_cat.csv`).
4. Generate explanations:

   ```python
   from explanation import Explanation
   explainer = Explanation(model_path='IDS1_model.h5', test_data=X_test_cnn, test_labels=y_test_cat)
   top_features, importance = explainer.explain_top_features(k=5)
   explainer.plot_top_features(top_features, importance)
   ```

### 4. Visualize Results

```python
import matplotlib.pyplot as plt

plt.barh(top_features, importance)
plt.xlabel("Feature Importance")
plt.title("Top 5 Features Affecting Prediction")
plt.show()
```

---

## Troubleshooting

* **Missing Libraries**

  ```bash
  pip install -r requirements.txt
  ```
* **xNIDS Import Error**

  ```python
  import sys
  sys.path.append('/path/to/code-xNIDS')
  ```
* **Model Loading Error**

  * Verify `IDS1_model.h5` is in the correct folder and not corrupted.
