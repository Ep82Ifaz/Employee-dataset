# Messy Employee Dataset: Cleaning, EDA and Performance Prediction

A Jupyter notebook (`messy-dataset.ipynb`) that loads a messy employee dataset, applies basic cleaning, explores it with five charts, and trains a Random Forest to predict `Performance_Score`.

**Headline result:** the model reaches **29.1% accuracy** on four roughly equal classes, where random guessing gives about 25%. In this dataset, the available columns carry little or no signal for performance (see [Results](#results) and [Limitations](#limitations)).

## Dataset

- **File:** `Messy_Employee_dataset.csv` (the notebook can read it directly from a `.zip` archive)
- **Size:** 1,020 rows × 12 columns
- **Source:** [Messy Employee Dataset on Kaggle](https://www.kaggle.com/datasets/desolution01/messy-employee-dataset)

| Column | Type | Notes |
|---|---|---|
| `Employee_ID` | text | Unique, format `EMP####` |
| `First_Name`, `Last_Name` | text | 8 distinct values each |
| `Age` | number | **211 missing (20.7%)**; only 4 distinct values (25, 30, 35, 40) |
| `Department_Region` | text | Department and state packed together (e.g. `DevOps-California`); 36 distinct labels |
| `Status` | text | Active, Inactive, Pending |
| `Join_Date` | text | `m/d/yyyy`, parsed to datetime in the notebook |
| `Salary` | number | **24 missing** |
| `Email` | text | Only 64 distinct values for 1,020 employees |
| `Phone` | integer | Stored as negative integers |
| `Performance_Score` | text | Target: Poor, Average, Good, Excellent |
| `Remote_Work` | boolean | True / False |

The raw data has no duplicate rows.

## What the notebook does

| Step | Cells | Description |
|---|---|---|
| 1. Setup | 0 | Imports libraries, fixes the random seed to 42, prints library versions |
| 2. Load | 1 | Finds the dataset in `/kaggle/input`, `/kaggle/working` or the current folder, either as a CSV or inside a `.zip` |
| 3. Inspect | 2 | Prints `info()`, missing values and duplicate count |
| 4. Clean | 3–4 | Parses `Join_Date`, converts `Age` and `Salary` to numeric, normalizes `Remote_Work`, drops `Employee_ID`, `First_Name`, `Last_Name`, `Email` and `Phone`. Result: 1,020 rows × 7 columns |
| 5. EDA | 5–9 | Age histogram, status count plot, salary box plot by `Department_Region`, join-year count plot, correlation heatmap |
| 6. Prepare features | 10–11 | Builds `Days_Since_Join`, drops rows with missing values, one-hot encodes the categorical columns, label-encodes the target |
| 7. Train | 12–13 | Stratified 70/30 split, Random Forest with 100 trees, classification report |
| 8. Evaluate | 14 | Confusion matrix heatmap |

## Results

**Data preparation**

- Rows before dropping missing values: 1,020
- Rows after: **787** (233 removed, **22.84%**)
- Feature matrix: 787 × 43 (after one-hot encoding)
- Train / test split: 550 / 237 rows, stratified on the target

**Random Forest (100 trees, seed 42)**

| Class | Precision | Recall | F1 | Test rows |
|---|---|---|---|---|
| Average | 0.27 | 0.24 | 0.25 | 66 |
| Excellent | 0.30 | 0.32 | 0.31 | 60 |
| Good | 0.28 | 0.31 | 0.29 | 61 |
| Poor | 0.33 | 0.30 | 0.32 | 50 |
| **Overall accuracy** | | | **0.2911** | 237 |

Every class scores between 0.25 and 0.32 F1, so the model is not strongly better at any one class.

## How to run

**1. Install the requirements** (Python 3.9 or newer):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

The notebook was run with pandas 2.3.3, numpy 2.1.3, matplotlib 3.10.0 and seaborn 0.13.2.

**2. Put the data next to the notebook**, either as the extracted CSV or as the zip:

```
project/
├── messy-dataset.ipynb
├── README.md
└── Messy_Employee_dataset.csv     (or the .zip that contains it)
```

**3. Open and run:**

```bash
jupyter notebook messy-dataset.ipynb
```

Then choose **Kernel → Restart & Run All**. On Kaggle, add the dataset to the notebook and run it as is.

## Limitations

These are the points to keep in mind when reading the results:

- **The model is close to chance.** 29.1% against a 25% chance level on 4 balanced classes is a small gap. The notebook has no baseline model, so this comparison is not shown in its output.
- **One split only.** Results come from a single 70/30 split of 237 test rows, so the accuracy can move by several points with a different seed. There is no cross-validation.
- **22.84% of the rows are dropped.** Rows with a missing `Age` or `Salary` are removed before modeling instead of being imputed.
- **Cleaning is light.** The notebook drops `Phone` and `Email` instead of cleaning them. It does not split `Department_Region` into department and state, so one-hot encoding creates 36 department-region columns.
- **`Age` is binned.** It has only 4 distinct values, so it behaves like a category and not a continuous age. This also limits the correlation heatmap, which covers only the numeric columns `Age` and `Salary`.
- **`Days_Since_Join` depends on today's date.** It is calculated from the day the notebook runs, so the raw values change daily. The relative ordering of employees stays the same.

## Ideas for next steps

- Add a majority-class baseline and 5-fold cross-validation to see whether the model beats chance reliably.
- Impute missing `Age` and `Salary` inside a pipeline, to keep all 1,020 rows.
- Split `Department_Region` into two columns.
- Test each feature statistically against `Performance_Score` (chi-square for categorical columns, Kruskal-Wallis for numeric ones) before modeling.
- Look for stronger performance drivers outside this dataset, such as manager ratings or goals met.