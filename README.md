# Lab10-ML
# Support Vector Machines – Iris Species Classification
**Course:** ARTI308 – Machine Learning

---

## Project Overview

This project builds a multi-class classification model using a **Support Vector Machine (SVM)** to predict the species of an iris flower. Given four morphological measurements, the model is trained, evaluated, and hyperparameter-tuned using GridSearchCV to identify the optimal decision boundary between species.

---

## Dataset

**Source:** [Iris Flower Dataset](http://en.wikipedia.org/wiki/Iris_flower_data_set) — introduced by Sir Ronald Fisher in 1936  
**Size:** 150 rows × 5 columns

| Feature | Description |
|---|---|
| `sepal_length` | Sepal length of the flower in cm |
| `sepal_width` | Sepal width of the flower in cm |
| `petal_length` | Petal length of the flower in cm |
| `petal_width` | Petal width of the flower in cm |
| `species` | **Target variable** – setosa, versicolor, or virginica |

---

## Techniques Applied

### Task 1 – Data Loading
- Loaded the dataset using `sns.load_dataset('iris')`
- Examined class distribution — 50 samples per species, perfectly balanced

### Task 2 – Exploratory Data Analysis (EDA)
- **Pairplot by species** – Plotted all feature combinations colored by species to identify separability patterns
- **KDE Plot (Setosa only)** – Kernel density estimate of sepal length vs. sepal width for the setosa species, showing its tight, well-defined cluster

### Task 3 – Data Preprocessing
- Separated features (`X`) from the target label (`y`)
- Split data into training (70%) and testing (30%) sets using `train_test_split` with `random_state=101`

### Task 4 – Model Training
- **Base SVC:** Trained a `SVC()` model with default hyperparameters on the training set

### Task 5 – Evaluation
- Generated a **Confusion Matrix** to examine per-class prediction accuracy
- Generated a **Classification Report** showing precision, recall, F1-score, and overall accuracy

### Task 6 – GridSearchCV Tuning
- Defined a `param_grid` over `C` ([0.1, 1, 10, 100]), `gamma` ([1, 0.1, 0.01, 0.001]), and `kernel` (['rbf'])
- Fitted `GridSearchCV` with 3-fold cross-validation to find the optimal parameter combination
- Re-evaluated the tuned model on the test set

---

## Results

| Model | Overall Accuracy | Best Parameters |
|---|---|---|
| Base SVC | 98% | Default |
| GridSearch SVC | 98% | C=1, gamma=0.1, kernel='rbf' |

---

## Conclusion

The SVM classifier performed exceptionally well on the Iris dataset, achieving 98% accuracy with only 1 misclassification out of 45 test samples. The EDA confirmed that *Iris setosa* is the most separable species — it forms a clearly distinct cluster from versicolor and virginica, especially in petal length and width. The GridSearchCV tuning did not improve beyond the base model's performance, which is expected given that the dataset is small, balanced, and linearly near-separable — conditions under which SVMs already perform near-optimally with default settings. The best confirmed parameters were `C=1, gamma=0.1, kernel='rbf'`.
