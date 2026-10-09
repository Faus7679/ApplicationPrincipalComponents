# ApplicationPrincipalComponents

## Telco customer churn PCA report

Open and run [`telco_customer_churn_pca_report.ipynb`](telco_customer_churn_pca_report.ipynb) from top to bottom to produce the analysis and figures requested in the assignment. The notebook reads the Excel workbook named in the assignment from its original Windows path. If you are running it elsewhere, update `DATA_PATH` in the data-loading cell to the location of your copy.

The notebook requires Python with `pandas`, `numpy`, `matplotlib`, and `openpyxl`. Install them with `python -m pip install pandas numpy matplotlib openpyxl` if needed. It excludes customer IDs and churn from the PCA features, standardizes the numeric customer attributes, and reports the rows excluded for missing or invalid values. Churn is used only to color the score plot when available.

The notebook contains the analysis code, generated plots, data-dependent findings, interpretation, and references. Results are calculated from the workbook when the notebook is run; no copy of the customer data is included in this repository.
