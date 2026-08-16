---
title: Genetic Algorithm Feature & Hyperparameter Optimizer
emoji: 🧬
colorFrom: red
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
license: apache-2.0
---

# Joint Feature Selection & Hyperparameter Optimization with GA

An interactive web application designed to run genetic algorithms for optimization of classifier models. The app jointly searches for:
1. **The optimal subset of features** (binary selection).
2. **The optimal hyperparameters** (KNN, SVM, Random Forest).

It is built with Streamlit and Plotly for deep visual insight into the GA's progression, fitness evolution, and classification gains.

## Features

- **Multi-Dataset Support:**
  - Built-in Breast Cancer classification dataset (Scikit-Learn).
  - OpenML Ionosphere dataset.
  - Custom file uploader to analyze your own CSV datasets.
- **Joint Optimization:** Run a Genetic Algorithm that operates across feature bits and hyperparameter chromosomes simultaneously.
- **Customizable GA Parameters:** Tweak population size, number of generations, and mutation rate on the fly.
- **Visual Performance Diagnostics:**
  - Dynamic Plotly comparison charts highlighting baseline vs. GA-optimized model accuracies.
  - Interactive convergence plots for best fitness over generations.
  - Breakdown table with selected features and specific hyperparameter combinations.

## Running Locally

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Hugging Face Spaces Deployment

This repository is pre-configured for deployment on **Hugging Face Spaces**.
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Select **Streamlit** as the SDK.
3. Push these project files (including `README.md`, `app.py`, and `requirements.txt`) to the Space repository.
