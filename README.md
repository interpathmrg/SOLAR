# Solar Radiation Analysis and Prediction Project

**Developer:** Miguel Gonzalez  
**Date:** June 10, 2025  
**Dataset Source:** [Kaggle Solar Radiation Dataset](https://www.kaggle.com/datasets/ibrahimkiziloklu/solar-radiation-dataset)

## Project Overview

This project performs comprehensive analysis and prediction of solar radiation data, specifically focusing on Global Horizontal Irradiance (GHI) prediction and energy production estimation for the "Planta Solar Girasol" solar farm. The project implements machine learning techniques to forecast solar energy production based on meteorological variables.

## Dataset Information

- **Source:** Solar Radiation Dataset from Kaggle
- **Primary Data File:** `2017.csv`
- **Key Variables:**
  - GHI (Global Horizontal Irradiance) - Target variable
  - DNI (Direct Normal Irradiance)
  - DHI (Diffuse Horizontal Irradiance)
  - Temperature
  - Relative Humidity
  - Wind Speed
  - Pressure
  - Timestamp components (Year, Month, Day, Hour, Minute)

## Project Structure

```
SOLAR/
├── data/
│   ├── 2017.csv
│   └── clima_futuro.csv
├── result/
│   ├── modelo_xgboost_GHI.joblib
│   ├── predicciones_GHI.csv
│   └── produccion_estimada_girasol.csv
└── scripts/
    ├── analisis.py
    ├── trainner.py
    ├── predecir_GHI.py
    └── calcular_energia_solar.py
```

## File Descriptions

### 1. `analisis.py` - Exploratory Data Analysis
**Purpose:** Initial data exploration and visualization

**Key Features:**
- Data loading and preprocessing
- Removal of low GHI values (< 5 W/m²) to filter dawn/dusk periods
- Statistical analysis and data quality assessment
- Visualization of daily GHI trends
- Hourly GHI distribution analysis using boxplots
- Correlation matrix analysis between meteorological variables

**Outputs:**
- Daily average GHI time series plot
- Hourly GHI distribution boxplot
- Correlation heatmap

### 2. `trainner.py` - Model Training and Evaluation
**Purpose:** Train XGBoost regression model for GHI prediction

**Key Features:**
- Data preprocessing and feature selection
- Temporal train-test split (80-20)
- XGBoost model training with hyperparameters:
  - n_estimators: 100
  - learning_rate: 0.1
  - max_depth: 6
- Model evaluation using MAE, RMSE, and R² metrics
- Feature importance analysis
- Error analysis by hour of day

**Model Performance Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared (R²) score

**Outputs:**
- Feature importance plot
- Actual vs Predicted scatter plot
- Residual error time series
- Hourly error analysis
- Trained model saved as `modelo_xgboost_GHI.joblib`

### 3. `predecir_GHI.py` - GHI Prediction
**Purpose:** Generate GHI predictions using the trained model

**Key Features:**
- Load pre-trained XGBoost model
- Process new meteorological data from `clima_futuro.csv`
- Generate GHI predictions
- Export predictions to CSV format

**Required Input Columns:**
- Temperature
- Relative Humidity
- DNI
- DHI
- Wind Speed
- Pressure

**Output:**
- `predicciones_GHI.csv` with predicted GHI values

### 4. `calcular_energia_solar.py` - Energy Production Calculation
**Purpose:** Calculate energy production for Planta Solar Girasol

**Solar Farm Specifications:**
- **Area:** 220 hectares (2,200,000 m²)
- **Efficiency:** 13% global efficiency
- **Time Resolution:** Hourly calculations

**Key Features:**
- Energy calculation: `Energy (kWh) = GHI × Area × Efficiency × Time_interval / 1000`
- Conversion to MWh for better visualization
- Hourly energy production visualization
- Export results to CSV

**Outputs:**
- Hourly energy production bar chart
- `produccion_estimada_girasol.csv` with energy estimates

## Installation and Dependencies

```bash
pip install pandas matplotlib seaborn xgboost scikit-learn joblib numpy
```

## Usage Instructions

### Step 1: Data Analysis
```bash
python analisis.py
```
Run this first to explore the dataset and understand data patterns.

### Step 2: Model Training
```bash
python trainner.py
```
Train the XGBoost model and evaluate its performance.

### Step 3: Generate Predictions
```bash
python predecir_GHI.py
```
Make predictions on new meteorological data.

### Step 4: Calculate Energy Production
```bash
python calcular_energia_solar.py
```
Estimate energy production for the solar farm.

## Model Features and Target

**Features (Input Variables):**
- Temperature
- Relative Humidity
- DNI (Direct Normal Irradiance)
- DHI (Diffuse Horizontal Irradiance)
- Wind Speed
- Pressure

**Target Variable:**
- GHI (Global Horizontal Irradiance)

## Data Preprocessing

- Removal of unnamed columns
- Filtering of low GHI values (< 5 W/m²)
- Timestamp creation and indexing
- Missing value handling
- Temporal data splitting for proper time series evaluation

## Model Architecture

**Algorithm:** XGBoost Regressor
- **Estimators:** 100 trees
- **Learning Rate:** 0.1
- **Max Depth:** 6
- **Random State:** 42 (for reproducibility)

## Energy Calculation Formula

```
Energy (kWh) = GHI (W/m²) × Area (m²) × Efficiency × Time_interval (hours) / 1000
Energy (MWh) = Energy (kWh) / 1000
```

## Results and Applications

This project enables:
- **Solar Resource Assessment:** Understanding solar irradiance patterns
- **Energy Production Forecasting:** Predicting energy output for solar farms
- **Operational Planning:** Supporting grid integration and energy trading decisions
- **Performance Monitoring:** Comparing actual vs predicted production

## Future Enhancements

- Integration of weather forecast APIs for real-time predictions
- Implementation of seasonal adjustment factors
- Addition of cloud cover and atmospheric variables
- Development of uncertainty quantification methods
- Creation of a web dashboard for real-time monitoring

## Technical Notes

- The model uses temporal splitting to avoid data leakage
- GHI values below 5 W/m² are filtered to focus on productive daylight hours
- The project assumes a global efficiency of 13% for the solar farm
- All calculations are performed at hourly resolution

## Contact Information

For questions or contributions, please contact Miguel Gonzalez.

---

*This project demonstrates end-to-end machine learning pipeline for solar energy forecasting, from data analysis to production estimation.*