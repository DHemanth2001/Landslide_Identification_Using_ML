Landslide Identification Using Machine Learning
Run Guide and Testing Instructions


OVERVIEW

This project is a two-phase machine learning system that identifies landslides from satellite images and predicts future landslide occurrences.

Phase 1 uses an AlexNet Convolutional Neural Network. It takes a satellite or aerial image as input and classifies it as either containing a landslide or not, along with a confidence percentage.

Phase 2 uses a Hidden Markov Model (HMM) trained on the NASA Global Landslide Catalog which contains 9,471 real-world events from 141 countries. Once Phase 1 detects a landslide, Phase 2 identifies the type of landslide, computes the probability of occurrence, predicts what type of landslide is most likely to happen in the next few events, and shows the peak risk month and historical statistics for the given country.


PREREQUISITES

Before running anything, you need to set up the conda environment. This only needs to be done once.

Step 1 — Create the conda environment

Run the following command in your terminal:

    conda create -n landslide-ml python=3.10 -y

Step 2 — Install dependencies

    conda activate landslide-ml
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install hmmlearn scikit-learn pandas numpy opencv-python albumentations tqdm joblib openpyxl torchinfo matplotlib seaborn

If the environment already exists from a previous session, skip the above and just activate it:

    conda activate landslide-ml

Step 3 — Navigate to the project directory

All commands must be run from inside this directory:

    cd /home/tnl3kor/hemanth-project/Landlside_Identification_Using_MachineLearning/project


NOTE ON RUNNING COMMANDS

All Python commands in this guide are prefixed with "conda run -n landslide-ml" to ensure they use the correct environment. If you have already activated the environment using "conda activate landslide-ml", you can drop the prefix and just run "python ..." directly.


SECTION 1 — RUNNING VIA JUPYTER NOTEBOOKS (RECOMMENDED FOR BEGINNERS)

The notebooks provide a step-by-step walkthrough of the entire project with explanations and visualizations. They are also compatible with Google Colab.

To launch the notebooks, run:

    conda run -n landslide-ml jupyter notebook notebooks/

This will open a browser window. Run the notebooks in the following order:

Notebook 00 — setup_and_data_download.ipynb
This notebook verifies that the dataset is in place and displays sample images from the training set. It shows examples of landslide and non-landslide patches side by side.

Notebook 01 — phase1_alexnet_training.ipynb
This notebook trains the AlexNet model on the HR-GLDD image dataset. It runs for 30 epochs by default, saving the best checkpoint automatically when validation accuracy improves. Loss and accuracy curves are plotted at the end.

Notebook 02 — phase1_evaluation.ipynb
This notebook evaluates the trained AlexNet on the held-out test set. It prints accuracy, precision, recall, and F1-score, and saves the confusion matrix and ROC curve to the plots directory.

Notebook 03 — phase2_hmm_training.ipynb
This notebook trains the HMM on the NASA Global Landslide Catalog CSV file. It loads 9,471 records, builds per-country temporal sequences, trains with Baum-Welch, and saves the model and encoder to the checkpoints directory.

Notebook 04 — phase2_hmm_evaluation.ipynb
This notebook evaluates the HMM. It shows the transition matrix, emission probabilities per hidden state, and runs predictions for several countries to demonstrate type classification and future forecasting.

Notebook 05 — full_pipeline_demo.ipynb
This notebook demonstrates the complete end-to-end pipeline. It takes test images, runs Phase 1 classification, and if a landslide is detected, runs Phase 2 to output the type, occurrence probability, future forecast, and country risk statistics.


SECTION 2 — RUNNING VIA COMMAND LINE

STEP 1 — TRAIN BOTH MODELS

To train both Phase 1 and Phase 2 from scratch in one command:

    conda run -n landslide-ml python pipeline/run_pipeline.py --mode train

This will first train the AlexNet model on the image dataset, then train the HMM on the NASA Global Landslide Catalog. Both checkpoints will be saved in the checkpoints directory when done.

If you only want to train Phase 1 (AlexNet) separately:

    conda run -n landslide-ml python phase1_alexnet/train.py

If you only want to train Phase 2 (HMM) separately:

    conda run -n landslide-ml python phase2_hmm/hmm_model.py


STEP 2 — PREDICT ON A SINGLE IMAGE

To run a full prediction on one image:

    conda run -n landslide-ml python pipeline/run_pipeline.py --mode predict --image /absolute/path/to/your/image.jpg --country India --forecast 3

Explanation of arguments:

--image is required. Provide the full absolute path to the image file. Supported formats are jpg, png, bmp, tif.

--country is optional. Provide a country name such as India, Nepal, Philippines, or United States. This enables Phase 2 to show historical risk statistics and peak risk month for that country. If omitted, Phase 2 still runs but without country-specific stats.

--forecast is optional. This controls how many future time steps to forecast. The default is 3. You can set it higher, for example --forecast 5, to get more future predictions.

Example output you will see:

    Phase 1 — LANDSLIDE (confidence: 52.2%)
    Phase 2 — Type: Landslide
             Occurrence probability: 39.6%
             Peak risk month: August
             Future forecast:
               Step 1: Landslide (prob=48.0%)
               Step 2: Landslide (prob=64.2%)
               Step 3: Landslide (prob=69.4%)

    LANDSLIDE DETECTED — Type: Landslide (occurrence probability: 39%)

If Phase 1 determines the image does not contain a landslide, Phase 2 is skipped and the output will say:

    NO LANDSLIDE DETECTED (confidence: 78%)


SECTION 3 — TESTING

HOW TO TEST PHASE 1 — ALEXNET IMAGE CLASSIFICATION

Run the evaluation script on the test set:

    conda run -n landslide-ml python phase1_alexnet/evaluate.py

This script loads the saved AlexNet checkpoint and runs it on all images in the data/processed/test directory, which contains 211 landslide images and 488 non-landslide images that were never seen during training.

It prints the following metrics:

Accuracy — overall percentage of correctly classified images. Expected value is around 79.5 percent.

Precision — out of all images predicted as landslide, how many actually are. Expected value is around 64.9 percent.

Recall — out of all actual landslide images, how many were correctly detected. Expected value is around 70.1 percent.

F1-Score — harmonic mean of precision and recall. Expected value is around 67.4 percent.

ROC-AUC — area under the receiver operating characteristic curve. Expected value is around 86.7 percent.

The confusion matrix image is saved to plots/confusion_matrix.png and the ROC curve is saved to plots/roc_curve.png.


HOW TO TEST PHASE 2 — HMM TYPE PREDICTION AND FORECASTING

Run the HMM prediction script:

    conda run -n landslide-ml python phase2_hmm/hmm_predict.py

This script loads the trained HMM and runs predictions for four countries: India, Nepal, Philippines, and United States. For each country it shows:

The current most likely landslide type based on the country's dominant historical pattern.

The occurrence probability as a percentage, computed from the HMM's log-likelihood score.

The peak risk month, which is the month historically having the most landslide events in that country.

Historical statistics including total events in the catalog, events per year, average fatalities, and dominant trigger (such as Heavy Rain or Earthquake).

A future forecast showing the most likely landslide type for the next 3 events along with their probabilities.


HOW TO TEST THE FULL PIPELINE TOGETHER

To run both phases and get a combined evaluation report:

    conda run -n landslide-ml python pipeline/run_pipeline.py --mode evaluate

This runs Phase 1 evaluation on the full test image set and prints all metrics, then computes the HMM log-likelihood across all 9,442 observations from the NASA catalog and prints the per-observation score.


QUICK TEST WITH ONE COMMAND

To quickly verify everything is working without manually specifying an image path, run this shell command which automatically picks the first available test image:

    IMAGE=$(ls data/processed/test/landslide/*.jpg | head -1) && conda run -n landslide-ml python pipeline/run_pipeline.py --mode predict --image "$IMAGE" --country Nepal


SECTION 4 — DATASET INFORMATION

The image dataset used for Phase 1 is the HR-GLDD (High-Resolution Global Landslide Dataset) from Zenodo (DOI: 7189381). It was downloaded as numpy array files and converted to JPEG images. The conversion script is located at data/convert_hrgldd_to_images.py.

Training images are in data/processed/train — 616 landslide images and 1574 non-landslide images.
Validation images are in data/processed/val — 158 landslide images and 392 non-landslide images.
Test images are in data/processed/test — 211 landslide images and 488 non-landslide images.

The dataset for Phase 2 is the NASA Global Landslide Catalog stored at data/excel/nasa_glc.csv. It contains 9,471 real-world landslide events recorded between 1988 and 2016 across 141 countries with columns for event date, country, landslide category, landslide trigger, landslide size, and fatality count.


SECTION 5 — SAVED MODEL FILES

The following files are already trained and saved in the checkpoints directory:

alexnet_best.pth — The trained AlexNet weights from epoch 20 with a validation accuracy of 78.73 percent.

hmm_model.pkl — The trained CategoricalHMM with 4 hidden states and 7 observation symbols.

hmm_encoder.pkl — The LabelEncoder for the 7 landslide types: Complex Landslide, Debris Flow, Earth Flow, Landslide, Mudslide, Rockfall, and Translational Slide.

These checkpoints are loaded automatically when running predictions or evaluations. You do not need to retrain unless you want to.


SECTION 6 — TROUBLESHOOTING

Problem: ModuleNotFoundError when running a script.
Solution: Make sure you are using the landslide-ml environment. Add "conda run -n landslide-ml" before your python command, or run "conda activate landslide-ml" first.

Problem: FileNotFoundError for image path.
Solution: Always use the full absolute path to the image file, not a relative path. For example use /home/username/project/data/processed/test/landslide/ls_test_0002.jpg instead of just ls_test_0002.jpg.

Problem: conda command not found.
Solution: Add conda to your PATH by running: export PATH="$HOME/miniconda3/bin:$PATH"

Problem: GPU not detected, using CPU.
Solution: Ensure you installed the CUDA-enabled version of PyTorch (cu124). If your CUDA version differs, replace cu124 with the appropriate version (e.g., cu118 for CUDA 11.8). Verify GPU availability with:

    conda run -n landslide-ml python -c "import torch; print(torch.cuda.is_available())"

If it prints False, check your NVIDIA drivers and CUDA toolkit installation.

Problem: HMM shows "Loaded 0 records".
Solution: The file data/excel/nasa_glc.csv may be missing or corrupted. Re-download it with this command:

    wget -O data/excel/nasa_glc.csv "https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/Global_Landslide_Catalog_Export_rows.csv"

Problem: AlexNet checkpoint not found.
Solution: The model has not been trained yet. Run the training command first:

    conda run -n landslide-ml python phase1_alexnet/train.py
