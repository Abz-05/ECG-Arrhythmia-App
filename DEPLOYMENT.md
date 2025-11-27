=# Deployment Guide: ECG Arrhythmia Classification App

This guide outlines the steps to deploy the ECG Arrhythmia Classification Streamlit app to **Streamlit Community Cloud** for free.

## Prerequisites

1.  **GitHub Account**: You need a GitHub account to host the repository.
2.  **Streamlit Account**: Sign up at [share.streamlit.io](https://share.streamlit.io/) using your GitHub account.
3.  **Git Installed**: Ensure Git is installed on your local machine.

## Step 1: Prepare the Repository

1.  **Verify Files**: Ensure your project structure looks like this:
    ```
    Signal Analysis/
    ├── app/
    │   └── streamlit_app_fixed.py
    ├── src/
    │   ├── dataset.py
    │   ├── evaluate.py
    │   ├── model.py
    │   ├── train.py
    │   └── utils.py
    ├── models/
    │   └── best_model_improved.pth  <-- IMPORTANT: Must be present
    ├── requirements.txt
    ├── .gitignore
    └── README.md (optional but recommended)
    ```

2.  **Check `requirements.txt`**:
    Ensure it contains all necessary libraries:
    ```text
    torch
    torchvision
    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn
    tqdm
    streamlit
    plotly
    ```

3.  **Check `.gitignore`**:
    Ensure `.gitignore` is configured to **include** the model files but exclude large data files and system files.
    ```gitignore
    # Ignore data
    data/
    *.csv

    # Ignore system files
    __pycache__/
    *.pyc
    .DS_Store
    .env
    .venv/
    
    # DO NOT IGNORE MODELS
    !models/
    !models/*.pth
    ```

## Step 2: Push to GitHub

1.  **Initialize Git** (if not already done):
    ```bash
    git init
    ```

2.  **Add Files**:
    ```bash
    git add .
    ```

3.  **Commit**:
    ```bash
    git commit -m "Prepare app for deployment"
    ```

4.  **Create a New Repository on GitHub**:
    *   Go to [github.com/new](https://github.com/new).
    *   Name it (e.g., `ecg-arrhythmia-app`).
    *   Make it **Public** (required for free Streamlit deployment).

5.  **Push Code**:
    Follow the instructions on GitHub to push your local code to the new repository.
    ```bash
    git remote add origin https://github.com/YOUR_USERNAME/ecg-arrhythmia-app.git
    git branch -M main
    git push -u origin main
    ```

## Step 3: Deploy to Streamlit Community Cloud

1.  **Log in** to [share.streamlit.io](https://share.streamlit.io/).
2.  Click **"New app"**.
3.  **Select Repository**:
    *   Choose your repository (`ecg-arrhythmia-app`).
    *   Branch: `main`.
    *   **Main file path**: `app/streamlit_app_fixed.py`.
4.  Click **"Deploy!"**.

## Step 4: Verify Deployment

1.  Wait for the app to build. You can see the logs in the bottom right corner.
2.  Once deployed, the app will open in a new tab.
3.  **Test**:
    *   Upload a sample CSV file (or use the provided sample data if you implemented that feature).
    *   Verify that the model loads correctly and predictions are generated.

## Troubleshooting

*   **Model Not Found**: If you see an error about the model file missing, ensure `models/best_model_improved.pth` was committed and pushed to GitHub. Check your `.gitignore`.
*   **Memory Issues**: The free tier has resource limits. If the app crashes, try reducing the model size or optimizing data loading.
*   **Dependency Errors**: Check the build logs. Ensure all packages in `requirements.txt` are correct and compatible.
