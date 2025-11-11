# TFG-Vulcan-Prediction
This is the repository of my Degree Final Project on the experimentation, evaluation and prediction of volcanic eruptions using AI.

## URLS:
- [Memory](https://drive.google.com/file/d/1PjFkvIIvh29cFnEbjCps4gafXO0FSlsx/view?usp=sharing)
- [Volcan CSV Kaggle](https://www.kaggle.com/competitions/predict-volcanic-eruptions-ingv-oe)
- [Mapbox](https://github.com/AitorRD/Volcano-map)

## Installation Manual
Before starting, make sure you have the following tools installed on your system:
- **Git**: to clone the repository. [Official page](https://git-scm.com/downloads)
- **Python 3.x**: the required programming language. [Official page](https://www.python.org/downloads/)
- **pip**: the Python package installer.

Once the prerequisites are met, proceed with the installation:

1. Run the following command to clone the repository from GitHub in your desired directory:
    ```bash
    git clone https://github.com/AitorRD/TFG-Vulcan-Prediction.git
    ```

2. Create a virtual environment using `venv`:
    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment (this step is optional but recommended):
    - On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

4. Run the following command to install all dependencies listed in the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

5. Download the competition data from the [Kaggle website](https://www.kaggle.com/competitions/predict-volcanic-eruptions-ingv-oe/data).

6. Unzip the downloaded file on your system.

7. Create the necessary directory to store the unzipped data:
    ```bash
    mkdir -p src/data/kaggle/input
    ```

8. Move the unzipped files to the created directory:
    ```bash
    mv path_to_unzipped_files/* src/data/kaggle/input/
    ```

9. Ensure the directory structure looks like this:
    ```
    Vulcan-Prediction/
    │
    ├── src/
    │   ├── data/
    │   │   └── kaggle/
    │   │       └── input/
    │   │           ├── test
    │   │           ├── train
    │   │           ├── sample_submission.csv
    │   │           └── train.csv
    ├── requirements.txt
    └── ...
    ```

