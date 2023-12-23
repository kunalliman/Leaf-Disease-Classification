# Leaf Disease Classification Project

This project aims to classify plant diseases using deep learning models. It utilizes the PlantVillage dataset from Kaggle to train models for identifying diseases in plant leaves.

## Getting Started

### Dataset
- Download the dataset from [Kaggle - PlantVillage Dataset](https://www.kaggle.com/arjuntejaswi/plant-village).
- Extract the downloaded data to a folder named "PlantVillage".

### Prerequisites
1. Activate the base environment or create a virtual environment.
   ```bash
    conda create -p venv python==3.8 -y
    ```
2. Install the required dependencies using [`requirements.txt`](requirements.txt).
    ```bash
    pip install -r requirements.txt
    ```
### Model Training Steps
- Refer to the presentation in the [PPT and Video](PPT and Video) folder for an overview of the project.
- Follow the notebooks in the following sequence:
  - [`Split_PlantVillage_Data.ipynb`](Split_PlantVillage_Data.ipynb): Split the dataset into train, test, and validation sets.
  - [`EDA.ipynb`](notebook_files/EDA.ipynb): Explore the data distribution.
  - Model Training Notebooks:
    - [`PepperLeaf_Model_Training.ipynb`](notebook_files/PepperLeaf_Model_Training.ipynb)
    - [`PotatoLeaf_Model_Training.ipynb`](notebook_files/PotatoLeaf_Model_Training.ipynb)
    - [`TomatoLeaf_Model_Training.ipynb`](notebook_files/TomatoLeaf_Model_Training.ipynb)
- Save the finalized models in the `saved_models` directory with naming conventions: `PepperLeaf_Model.h5`, `PotatoLeaf_Model.h5`, `TomatoLeaf_Model.h5` in their respective plant folder.

### User Interface
- The `ui` directory contains static and template folders for a user-friendly web interface of the app.
- Resources folder includes the [`model_operations.py`](Resources/model_operations.py) module with necessary functions used in [`app.py`](app.py).
- Test your code using [`test.py`](Resources/test.py) in Resources before running it in the app.

#### Click the User Interface image to see the Demo Video.
<div style="display: flex; justify-content: space-between;">
    <a href="https://youtu.be/pxD55ha6eMA" target="_blank">
        <img src="PPT and Video/home_page.png" alt="User Interface Demo Video" width="400"/>
    </a>
    <a href="https://youtu.be/pxD55ha6eMA" target="_blank">
        <img src="PPT and Video/predictor_page.png" alt="Second Image" width="400"/>
    </a>
</div>



### Running the App
- Activate conda environment `conda activate` with requirements installed already. 
- Run the `python app.py` code in the terminal to start the web page. Access the app by copying the provided localhost link.

## Future Scope
- Explore alternative models beyond CNNs.
- Expand the project to include more plant species and fine-tune models for specific plants.
- Enhance production-level deployment and incorporate CI/CD pipelines for automated testing and deployment.
- Collaborate with experts for domain-specific insights and continuous data collection.

---

This README file provides a step-by-step guide to setting up the project, running the code, and outlines future directions for the project's expansion and improvement.
