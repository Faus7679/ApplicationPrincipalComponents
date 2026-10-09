# ApplicationPrincipalComponents

This project analyzes the Telco Customer Churn data with Principal Component
Analysis (PCA). The repository does not include the source data, so keep the
Excel workbook locally and pass its path to the script.

## Run the analysis

Install the required scientific Python packages, then run:

```bash
python -m pip install numpy pandas matplotlib scikit-learn openpyxl
python pca_telco_analysis.py --data "/path/to/CST-570-RS-WAFn-UseC-Telco-Customer-Churn.xlsx"
```

The script also accepts a CSV file. It writes `explained_variance.png`,
`pca_projection.png`, and `pc1_loadings.png` to `pca_outputs/`, and prints the
numeric evidence used in the answers below.

## Answers to questions 1–5

### 1. Why is PCA useful for customer churn analysis?

Telco data mixes numeric measurements (such as tenure and charges) with many
categorical service and contract variables. One-hot encoding can therefore
create a wide design matrix, and related variables can be strongly correlated.
PCA replaces those correlated columns with orthogonal components ordered by
variance. This can reduce the number of inputs, make two-dimensional
visualization possible, reduce multicollinearity, and make a downstream model
faster. PCA is not a churn predictor by itself: it is an unsupervised
representation of customer features, so the `Churn` label is excluded while
components are fitted.

### 2. How was PCA applied and how many components should be retained?

The analysis removes identifiers, converts `TotalCharges` to numeric, imputes
numeric missing values with the median, replaces missing categorical values,
one-hot encodes categorical columns, and standardizes every encoded feature.
Standardization is essential because otherwise a large-unit variable could
dominate the covariance structure. The script fits PCA to these standardized
features and prints the variance ratio for each component. The
`explained_variance.png` scree/cumulative plot supports an evidence-based
choice: retain the smallest number of components that meets the desired
variance threshold (90% is shown), while also reporting the first three
components for the requested compact representation. There is no honest
fixed number independent of the input file; the printed values are the
dataset-specific answer.

### 3. What does the first-two-component projection show?

`pca_projection.png` plots every customer using PC1 and PC2 and colors points by
the original churn label. Overlap means churn is not perfectly separable by
the two largest variance directions; separated regions suggest that those
directions contain churn-related structure. PCA maximizes total feature
variance, not class separation, so a visually mixed plot does not prove that
churn is unpredictable. The plot is descriptive, while the hold-out metrics
printed by the script provide a quantitative check.

### 4. Which features contribute most to PC1?

The signed loadings in the console and `pc1_loadings.png` identify the
encoded features with the largest absolute contributions to PC1. Features with
positive loadings move customers in one direction and negative loadings move
them in the opposite direction; the sign itself is arbitrary, so the
magnitudes are the primary evidence. Groups of large charge/tenure loadings
can be interpreted as an overall customer-value or service-engagement axis,
whereas large contract or service indicators may describe a contract/service
mix. The interpretation must follow the actual top loadings printed for the
provided workbook rather than assuming a feature is important in advance.

### 5. Does PCA improve churn-model performance?

The script uses the same stratified 75/25 hold-out split and random seed to
compare logistic regression on all standardized encoded features with
logistic regression on the first three principal components. It reports
accuracy, balanced accuracy, F1, and ROC AUC. If the compact model is close to
the full model, PCA achieves useful compression with lower dimensionality and
less multicollinearity. If it is worse, the omitted components contain
predictive information even if they explain less total variance. Thus PCA is a
trade-off, not a guarantee of better accuracy: fewer features improve speed
and stability, but transformed components are less directly interpretable and
may discard low-variance information that is important for churn.
