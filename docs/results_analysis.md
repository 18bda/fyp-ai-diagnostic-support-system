# Results Analysis

The experimental results show that ensemble-based models such as Random Forest, Extra Trees, and HistGradientBoosting achieved the highest accuracy across the datasets.

HistGradientBoosting achieved the best performance, with accuracy close to 0.997, indicating strong predictive capability on the heart disease dataset.

Tree-based models consistently outperformed simpler models such as Logistic Regression and K-Nearest Neighbours. This suggests that the dataset contains non-linear relationships between features.

Support Vector Machines and KNN showed lower performance, which may be due to sensitivity to feature scaling and data distribution.

The use of repeated stratified cross-validation ensures that the results are reliable and not dependent on a single train-test split.

In comparison, the symptom-based dataset produced near-perfect accuracy due to the structured nature of symptom patterns, whereas the heart disease dataset represents a more realistic and challenging prediction task.