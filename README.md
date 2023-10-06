# Prediction-of-COVID-19-Using-Machine-Learning-Models

# Overview
This project focuses on utilizing machine learning for predicting COVID-19 diagnoses using a dataset from Hospital Israelita Albert Einstein, São Paulo, Brazil. The experiments involved Cross-Validation and Holdout methods, evaluating five models. XGBoost and Random Forest demonstrated superior performance with 91% accuracy, emphasizing the significance of features like Basophils and Patient Age Quantile. The discussion delves into model strengths and limitations, offering valuable insights for ongoing research in enhancing COVID-19 diagnosis through machine learning.

## Dataset Description
The dataset for this report, sourced from Kaggle: https://www.kaggle.com/datasets/einsteindata4u/covid19 , originates from patients receiving medical care at the Hospital Israelita Albert Einstein in São Paulo, Brazil. The dataset consists of confidential samples collected through RT-PCR tests for SARS-CoV-2 and other laboratory tests, with 5,644 rows and 111 columns. 

## Data Pre-processing
### Dropping Unnecessary Columns
Certain columns (column1, column2, column3) were deemed unnecessary for analysis and were dropped from the dataset.

### Handling Missing Values
Approximately 50% of the columns had over 90% missing values. These columns were removed, resulting in a new dataset of 5,644 rows and 35 columns 


### Filtering Data
The data was filtered to include positive SARS-CoV-2 exam results and non-null values in the 'Hematocrit' or 'Urine - Density' columns. This resulted in a focused dataset of 1,091 rows and 35 columns.

### Encoding Categorical Features
One-third of the variables in the dataset were categorical. LabelEncoder was used to convert them into numerical values for consistency.

### Data Balance
After preprocessing, the dataset contains 1,091 rows and 35 columns with a balanced distribution.

## Machine Learning Models
The report evaluates five classifiers: XGBoost, Random Forest (RF), Decision Tree (DT), Logistic Regression (LR), and K-Nearest Neighbors (KNN).

# Experiment-1
### Cross-Validation Method
In Experiment-1, we employed the Cross-Validation (CV) method, a machine learning technique assessing model performance on unseen data. The dataset was divided into ten folds, with k-1 folds used for training and the remaining one for validation in each iteration. This technique helps provide a more robust evaluation of the models.

### Hyperparameters Optimization
To determine the best combination of hyperparameters for the ML models (XGBoost, Random Forest, Decision Tree, Logistic Regression, K-Nearest Neighbors), a grid search technique with cross-validation was utilized. Optimal hyperparameters were identified, and the models were trained using these parameters.

### Result Evaluation
Performance outcomes were obtained for different CV fold values, and the classifiers were compared based on metrics such as F1-score, testing accuracy, precision, and recall. The ROC-AUC curve was also used to assess the models' performance.

# Experiment-2
### Holdout Method with Result Evaluation
In Experiment-2, the Holdout method was employed, where 20% of the data was reserved for testing, and 80% was used for training. The ML classifiers were evaluated based on precision, recall, F1-score, accuracy, AUC, and confusion matrix. This approach provides insights into how the models generalize to new, unseen data.

### Important Features using SHAP Values
The SHapley Additive exPlanations (SHAP) values were utilized to identify the most informative predictors contributing significantly to the decision-making process of the classifiers. This analysis helps interpret and understand the features that play a crucial role in the models' predictions.

# Discussion and Conclusion
This study assessed five ML models for COVID-19 diagnosis with the goal of assisting medical professionals in choosing the best models for particular applications. According to our findings, both XGBoost and RF managed to attain a 91% accuracy rate, however RF had a higher recall for true positives than XGBoost, at 95% as opposed to 93%. Both models exhibited high AUC values, with RF performing somewhat better score of 95.7% with cross-validation. Again, applying 20% test data samples in the holdout approach, both models had the highest AUC of 96%, but RF had the highest accuracy and F1-Score at 92%. Our research suggests that RF could be a better option for this classification challenge. The top two crucial features for the model's prediction are Basophils and Patient Age Quantile. However, one limitation of this report is that the dataset only includes patients who had blood or urine testing, which may not adequately represent the total population. The accuracy of the diagnosis may also be impacted by other demographic and clinical parameters that weren't considered in the model's performance. Therefore, further investigation is necessary to establish the influence of other clinical factors on ML models' performance in COVID-19 diagnosis.
