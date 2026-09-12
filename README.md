# House Price Prediction using Linear Regression

**MainCrafts Technology — AI/ML Internship, Task 1**

## 📋 Objective

Build and evaluate a Linear Regression model on the California Housing dataset, covering the complete machine learning workflow: data loading, exploratory data analysis (EDA), preprocessing, model training, evaluation, and reporting.

## 📊 Dataset

The [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (built into scikit-learn) — 20,640 rows, 9 columns, describing housing characteristics across California districts. Goal: predict the median house value (`MedHouseVal`) for each district.

**Features used:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

## 🛠️ Tech Stack

- Python
- pandas, numpy
- scikit-learn (LinearRegression, train_test_split, metrics)
- matplotlib, seaborn (visualization)
- ipywidgets (interactive UI)

## 🔍 Workflow

1. **Data Loading** — Loaded the California Housing dataset via `sklearn.datasets.fetch_california_housing`
2. **Exploratory Data Analysis (EDA)** — Checked for missing values, examined feature distributions, and built a correlation heatmap
3. **Data Preparation** — Split data into training (80%) and test (20%) sets
4. **Model Training** — Trained a `LinearRegression` model
5. **Evaluation** — Assessed performance using MAE, RMSE, and R² score
6. **Visualization** — Plotted Actual vs Predicted values and residual diagnostics
7. **Deployment** — Saved the trained model as a pickle file and built an interactive prediction UI using `ipywidgets`

## 📈 Results

| Metric | Value | Meaning |
|--------|-------|---------|
| MAE | 0.533 | Average absolute prediction error (~$53,300) |
| RMSE | 0.746 | Root mean squared error (~$74,600) |
| R² Score | 0.576 | Model explains ~57.6% of the variance in house prices |

**Key finding:** Median Income (`MedInc`) is by far the strongest predictor of house value (correlation = 0.688), consistent with real-world intuition.

## 🖥️ Interactive Prediction UI

An interactive widget-based UI was built to let users enter neighborhood details (income, house age, rooms, location, etc.) and instantly get a predicted median house price.

**Example:** For median income $50,000, house age 25 years, 6 average rooms, 1 average bedroom, population 1,000, 3 people per household, near Los Angeles → predicted value ≈ **$239,840.67**

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `task1_ml_linear_regression.ipynb` | Full Jupyter Notebook with code, plots, and comments |
| `House_Price_Estimator.docx` | Detailed project report (EDA, model, metrics, visualizations, conclusions) |
| `house_price_model.pkl` | Saved trained Linear Regression model |

## 💡 Improvement Ideas

- Feature scaling (StandardScaler) for more comparable coefficients
- Feature engineering (e.g. rooms-per-household, distance to city center)
- Non-linear models (Random Forest, Gradient Boosting) — typically reach R² of 0.75–0.85+ on this dataset
- Handling the artificially capped $500,000 house values, which distort error metrics
- Cross-validation instead of a single train/test split

## 🙋 About

This project was completed as part of the AI/ML Internship program at **MainCrafts Technology**.

- Website: [maincrafts.com](https://maincrafts.com)
