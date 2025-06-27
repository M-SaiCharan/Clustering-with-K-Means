#  Clustering with K-Means

This project demonstrates how to perform customer segmentation using **K-Means Clustering** on the **Mall Customer Segmentation** dataset. It includes dimensionality reduction using **PCA**, cluster evaluation using the **Elbow Method** and **Silhouette Score**, and visualization of the clusters.

---

##  Dataset

**File:** `Mall_Customers.csv` 

The dataset includes the following columns:
- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1–100)`

---

## libraries used
pandas: For reading the CSV file and handling tabular data (dataframes).

numpy: For numerical operations and array manipulations.

matplotlib: For creating plots such as the Elbow Method and cluster visualizations.


##  Objectives

1. Load and visualize the dataset
2. Perform PCA (optional) to reduce dimensions to 2D for visualization
3. Apply K-Means clustering
4. Use the Elbow Method to determine the optimal number of clusters
5. Visualize clusters and centroids
6. Evaluate clustering performance using Silhouette Score

---

##  Requirements

Install the required Python packages:

```bash
pip install pandas numpy matplotlib scikit-learn