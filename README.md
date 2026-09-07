# Spaceship Titanic - Machine Learning Pipeline

This repository contains my end-to-end machine learning solution for the Kaggle competition **[Spaceship Titanic](https://www.kaggle.com/c/spaceship-titanic)**. The goal of this project is to predict whether a passenger was transported to an alternate dimension during the spaceship's collision with a spacetime anomaly.

---

##  Project Overview & Pipeline Steps

The project follows a robust, production-ready data science pipeline built in Python:

1. **Data Preprocessing & Imputation:**
   * Handled missing values using `SimpleImputer` (median for numerical columns, most frequent for categorical columns).
   * Cleaned and optimized boolean and categorical features.

2. **Feature Engineering:**
   * Extracted useful signals from composite columns (e.g., splitting the `Cabin` feature into `Deck`, `Num`, and `Side`).
   * Addressed specific domain logic (e.g., resetting luxury spending features to zero for passengers in `CryoSleep`).

3. **Encoding:**
   * Converted categorical variables into numeric formats using `OneHotEncoder` with `handle_unknown='ignore'`.

4. **Model Training & Validation:**
   * Evaluated baseline models like **Random Forest Classifier**.
   * Implemented and optimized **XGBoost Classifier** (`XGBClassifier`) for enhanced performance, achieving a validation accuracy of **~80.1%** and a public leaderboard score of **~0.795**.

---

##  Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost

---

##  Repository Structure

* `spaceship-titanic-notebook.ipynb`: The complete Jupyter Notebook containing data exploration, feature engineering, model training, and prediction.
* `submission.csv`: The final generated prediction file ready for Kaggle submission.
