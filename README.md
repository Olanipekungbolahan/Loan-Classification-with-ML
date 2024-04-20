# Loan-Classification-with-ML
# Machine Learning Model Deployment with Flask and Docker

This repository contains code for building, training, and deploying machine learning models using Random Forest, XGBoost, and Deep Learning algorithms. The models are trained on a loan dataset after data analysis, exploration, and preprocessing steps. The trained models are then integrated into a Flask web application and deployed using Docker in a cloud environment.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Machine learning models have been built using Random Forest, XGBoost, and Deep Learning algorithms to solve a specific problem. This README provides a detailed guide on how to reproduce the project and deploy the models into a cloud environment.

## Project Structure

The project is structured as follows:

- `data/`: Contains the dataset used for training the models.
- `models/`: Contains the trained machine learning models saved as pickle files.
- `app/`: Contains the Flask web application for model deployment.
- `scripts/`: Contains scripts for data preprocessing, model training, and evaluation.
- `Dockerfile`: Specifies the Docker image configuration.
- `requirements.txt`: Lists all the Python dependencies required for the project.
- `README.md`: The file you're currently reading.

## Requirements

To run the project, you need to have the following dependencies installed:

- Python (>=3.12.2)
- Flask
- scikit-learn
- XGBoost
- TensorFlow (for Deep Learning)
- Docker

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your_username/your_project.git
cd your_project
