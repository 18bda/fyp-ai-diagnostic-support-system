# Symptom baseline verification summary

## Why verify?
The baseline symptom classifier achieved 100% accuracy, which can indicate either data leakage or a trivial dataset.  
This verification pack checks for common leakage indicators.

## Results (Training.csv)
- Dataset: 4,920 rows, 132 binary symptom features, 41 classes
- Duplicate symptom vectors: none detected (all rows unique)
- Train/test overlap of identical symptom vectors: 259
- Hold-out accuracy (stratified split): 1.0000
- 5-fold CV accuracy: mean=1.0000, std=0.0000
- Label-shuffled accuracy: 0.0081 (near chance)

## Interpretation
The label-shuffled test being near chance shows the model is not memorising labels through leakage.
However, the consistently perfect performance suggests the dataset may be inherently easy/trivial,
so a harder second dataset will be introduced for more meaningful comparison across multiple algorithms.

