# Multimodal-Sentiment-Analysis-System-for-War-News
A small project involving scrapping X data, websites and doing sentimental analysis of it using DistilBERT, ResNet-50 and VADER

## The Architecture
The core of this is a custom PyTorch "Fusion-Net" that forces two distinct intelligence streams to converge into a single decision-making head:

* **Linguistic Stream:** Employs **DistilBERT** to extract 768-dimensional semantic embeddings from news content and social media (X) discourse.
* **Visual Stream:** Utilizes a specialized **ResNet-50** (with the final classification layer decapitated) to extract 2048-dimensional feature vectors from battlefield and situational imagery.
* **The Fusion Layer:** Features are concatenated and passed through a series of Fully Connected (FC) layers with **Dropout (0.3)** to prevent overfitting on the high-frequency noise of social media data.

## Pipeline & Usage
The repository is structured as a sequential pipeline. Ensure your `master_images/` folder and `final_war_dataset_PERFECT.json` are in the root directory.

1.  **Data Preprocessing:** Standardize the integrated News + X dataset.
    ```bash
    python preprocess_data.py
    ```
2.  **Auto-Labeling:** Generate ground-truth labels using VADER-assisted logic.
    ```bash
    python auto_label.py
    ```
3.  **Feature Extraction:** Convert raw data into high-dimensional `.pt` tensors (BERT + ResNet).
    ```bash
    python extract_features.py
    ```
4.  **Neural Training:** Train the Multimodal Fusion-Net.
    ```bash
    python train_model.py
    ```
5.  **Inference:** Deploy the model for real-time testing.
    ```bash
    python inference.py
    ```
