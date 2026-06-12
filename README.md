Multi-Person 3D Human Pose Tracking Using Multi-View 2D Detections
==================================================================

## Key Project Deliverables

For evaluation purposes, the most important files are:

* **Source Code:** `code/` - Contains all the source code for the project, including data exploration, training, and evaluation.
* **Project Report:** `report/main.pdf` - The comprehensive project report.
* **Final Presentation Slides:** `presentation/main.pdf` - The slides for the final project presentation.

## Project Overview

<!--TODO -->

### Project Structure

<!--TODO -->

## Setup, Installation, Usage

Before starting, you should download the repository locally by running the following commands:

```bash
git clone https://github.com/rolandbernard/cv-project
cd cv-project
```

### Setup

To set up and run this project, follow these steps after downloading the project:

1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
2.  **Install dependencies:**
    The project relies on a set of Python libraries. Install them using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` includes specific versions for reproducibility. Ensure your environment is compatible or remove the versions.*

    Alternatively, run the following, which is also the first code cell in `0.setup.py`:
    ```bash
    pip install numpy pandas scipy pillow opencv-python opencv-contrib-python torch torchvision ultralytics matplotlib seaborn pyvista[jupyter] jupyter ipywidgets gdown einops git+https://github.com/microsoft/MoGe.git
    ```
3. **Download the Datasets:**
    To download the datasets, and install missing dependencies (if not done in the previous step), simply run all cells in the `0.setup.py` notebook. It will automatically download the datasets and extract them into the `code/data` directory.

### Usage

The project is divided into multiple Jupyter Notebooks in the `code/notebooks/` directory.
* For initial setup of dependencies and to download the datasets, take a look at `0.setup.ipynb`.
* `1.exploration.ipynb` contains code for performing some initial data exploration.
* `2.a.train.kalman.ipynb` contains the code for training the Kalman filter parameters.
* `2.b.train.yolo.ipynb` contains the code for training the custom YOLO26 keypoint head.
* `3.evaluation.ipynb` includes the code for running evaluation using the test set.

In case you want to inspect or modify the tracking algorithm, `tracker.py` contains the main entry point, with the Kalman filter components in `kalman.py` and the detection parts in `detect.py`. Note that the implementation can be configured in different ways, e.g., changing the physics model or the detection network.
