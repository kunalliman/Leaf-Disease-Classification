# Plant Disease Classification Project

This project aims to classify plant diseases using deep learning models. It utilizes the PlantVillage dataset from Kaggle to train models for identifying diseases in plant leaves.

## Getting Started

### Dataset
- Download the dataset from [Kaggle - PlantVillage Dataset](https://www.kaggle.com/arjuntejaswi/plant-village).
- Extract the downloaded data to a folder named "PlantVillage".

### Prerequisites
- Refer to the presentation in the "PPT and Video" folder for an overview of the project.
- Follow the notebooks in the following sequence:
  - `Split_PlantVillage_Data.ipynb`: Split the dataset into train, test, and validation sets.
  - `EDA.ipynb`: Explore the data distribution.
  - Model Training Notebooks:
    - `PepperLeaf_Model_Training.ipynb`
    - `PotatoLeaf_Model_Training.ipynb`
    - `TomatoLeaf_Model_Training.ipynb`
- Save the finalized models in the `saved_models` directory with naming conventions: `PepperLeaf_Model.h5`, `PotatoLeaf_Model.h5`, `TomatoLeaf_Model.h5`.

### User Interface
- The `ui` directory contains static and template folders for a user-friendly web interface of the app.
- Resources folder includes the `model_operations.py` module with necessary functions used in `app.py`.
- Test your code using `test.py` in Resources before running it in the app.

### Running the App
1. Activate the base environment or create a virtual environment.
2. Install the required dependencies using `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
3. Run the `app.py` code in the terminal to start the web page. Access the app by copying the provided localhost link.

## Future Scope
- Explore alternative models beyond CNNs.
- Expand the project to include more plant species and fine-tune models for specific plants.
- Enhance production-level deployment and incorporate CI/CD pipelines for automated testing and deployment.
- Collaborate with experts for domain-specific insights and continuous data collection.

---

This README file provides a step-by-step guide to setting up the project, running the code, and outlines future directions for the project's expansion and improvement. Adjust the instructions as needed and add any additional details or considerations specific to your project.