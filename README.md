### Requirements

Install `conda`

#### Install MLFlow locally in a virtual env
```
python3 -m venv venv
source venv/bin/activate
pip install mlflow
```

Install `wandb`

#### Must have account on https://wandb.ai

# Build a Reproducible Model Workflow - Exercises

This repo contains the code for demos, exercises, and exercise solutions.

This repository organizes the code by the lessons that they are used in. 
Each set of code is located in their respective lessons.

Please note that certain instructions for each exercise, as well as any 
relevant environment setup, are only provided within the Udacity classroom.

## Example:
All lesson 2 files are in `/lesson-2-data-exploration-and-preparation/`.

This directory contains: `demo`, `exercises`, with the `exercises` directory 
organized by the exercise number, and therein containing an exercise `README.md` 
file and `starter` and `solution` directories.

# Notes:
## How this looks in real pipelines
- **What companies actually use**
  - **PyTorch**
    - Dominant for neural networks
    - **Used at Meta, OpenAI, Tesla, Uber, Netflix, Stripe, etc.**
    - First-class support for:
      - GPUs / distributed training
      - Custom architectures
      - Research → production workflows
    - De facto industry standard in 2025
    
  - **scikit-learn**
    - Not used for neural networks in production
    - Used for:
      - Linear / logistic regression
      - Random forests, GBMs
      - Clustering, PCA
      - Feature preprocessing
    - Its neural net module (MLPClassifier) is mostly educational / legacy

## Very common setup:
- **scikit-learn** → data prep, baselines, classical models
- **PyTorch** → deep learning models
- **XGBoost** / LightGBM → tabular production work
- ONNX / TorchScript / Triton → serving

- What about TensorFlow?
  - Still used (especially legacy systems)
  - Less common for new projects
  - **PyTorch** has won mindshare and hiring

## Bottom line

- Neural networks in production → **PyTorch**
- Classical ML in production → **scikit-learn** + **XGBoost**
- **scikit-learn** NN ≠ industry practice