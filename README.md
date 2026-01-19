# Crop Prediction with Machine Learning

A machine learning project that helps farmers optimize crop selection by predicting the best crop to plant based on soil composition measurements.

## Overview

This project builds a multi-class classification model to predict optimal crop selection using soil nutrient levels. By analyzing nitrogen (N), phosphorous (P), potassium (K), and pH measurements, the model identifies which crop will yield the best results for a given field's soil conditions.

## Dataset

The dataset (`soil_measures.csv`) contains 2,200+ soil samples with the following features:
- **N**: Nitrogen content ratio in the soil
- **P**: Phosphorous content ratio in the soil  
- **K**: Potassium content ratio in the soil
- **pH**: pH value of the soil
- **crop**: Target variable (22 different crop types)

## Methodology

1. **Data Loading & Exploration**: Loaded and examined soil measurements dataset
2. **Feature Engineering**: Isolated individual soil metrics to assess predictive power
3. **Model Training**: Trained separate logistic regression models for each soil feature
4. **Performance Evaluation**: Compared weighted F1 scores to identify the most important feature
5. **Feature Selection**: Determined that **Potassium (K)** is the strongest single predictor

## Key Results

| Feature | F1 Score |
|---------|----------|
| Potassium (K) | **0.126** |
| Phosphorous (P) | 0.098 |
| Nitrogen (N) | 0.086 |
| pH | 0.034 |

**Key Finding**: Potassium levels are the most predictive single feature for crop selection, achieving an F1 score of 0.126.

## Technologies Used

- **Python 3.8+**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning models and evaluation metrics
  - LogisticRegression (multinomial)
  - train_test_split
  - f1_score (weighted)
  - StandardScaler

## Installation & Usage

```bash
# Install required packages
pip install pandas scikit-learn

# Run the analysis
python crop_prediction.py
```

## Project Structure

```
├── soil_measures.csv          # Dataset
├── crop_prediction.py         # Main analysis script
└── README.md                  # Project documentation
```

## Future Improvements

- Implement ensemble models (Random Forest, Gradient Boosting) for improved accuracy
- Add feature interactions and polynomial features
- Create data visualizations showing crop distributions by soil type
- Develop a complete multi-feature model combining all soil measurements
- Add cross-validation for more robust performance estimates

## Applications

This type of analysis can help:
- Farmers make data-driven crop selection decisions
- Agricultural consultants provide evidence-based recommendations
- Reduce costs associated with soil testing by identifying the most critical metrics
- Optimize crop yields through better soil-crop matching
