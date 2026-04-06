# Multimodal-Sentiment-Analysis-System-for-War-News
A small project involving scrapping X data, websites and doing sentimental analysis of it using DistilBERT, ResNet-50 and VADER

## The Architecture
The core of this system is a custom PyTorch **"Fusion-Net"** that forces two distinct intelligence streams to converge into a single decision-making head:

* **Linguistic Stream:** Employs **DistilBERT** to extract 768-dimensional semantic embeddings from news content and social media (X) discourse.
* **Visual Stream:** Utilizes a specialized **ResNet-50** (with the final classification layer decapitated) to extract 2048-dimensional feature vectors from battlefield and situational imagery.
* **The Fusion Layer:** Features are concatenated and passed through a series of Fully Connected (FC) layers with **Dropout (0.3)** to prevent overfitting on the high-frequency noise of social media data.

## Pipeline & Usage
The repository is structured as a sequential pipeline. To replicate the dataset and train the model, ensure your environment is set up and follow these steps in order:

1.  **Ingest Social Data:** Execute the `twitter_scrape.py` file to collect real-time X discourse.
2.  **Ingest Global News:** Open and execute the cells in `Webscrapping.ipynb` to capture data from 25+ international news agencies (Reuters, Al Jazeera, etc.).
3.  **Data Engineering:** Execute the cells in `Master_Dataset_Creation_Program.ipynb`. This notebook:
    * Consolidates fragmented JSON batches into the master file.
    * Handles **Path Neutralization** and image consolidation into the `master_images/` folder.
    * Performs deduplication to ensure a high-integrity dataset of 1,260+ unique entries.
4.  **Model Training:** Finally, execute the cells in `Training.ipynb`. This notebook performs:
    * **Auto-Labeling:** VADER-assisted sentiment categorization.
    * **Feature Extraction:** Generates BERT and ResNet feature tensors.
    * **Neural Training:** Optimizes the Fusion-Net and generates the `multimodal_model.pth` weights.

*(Note: The `master_images/` folder and `final_war_dataset_PERFECT.json` are generated dynamically by the pipeline and are excluded from the initial repo due to size constraints.)*
