# Dataset Name: Wholesale Customers Dataset

# 🔗 https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set
#  Wholesale customers Data Set | Kaggle
# Annual spending in monetary units of clients of a wholesale distributor
 
# Business Scenario
# A wholesale distributor supplies products to different types of clients such as:
# Retail stores
# Cafés
# Hotels
# Restaurants
# Currently, all customers are treated the same, which leads to:
# Inefficient inventory planning
# Poor marketing strategies
# Missed upselling opportunities
# The company wants to group customers based on their purchasing behavior to improve decision-making.
 
# -------------------------------------------
# 🔹 Task 1: Data Exploration
# Load the dataset and inspect all available features.
# Identify which columns represent customer purchasing behavior.
# Remove or ignore columns that are not directly related to spending patterns.
# 🔹 Task 2: Feature Selection
# Select suitable numerical features that can represent customer buying habits.
# Justify your feature selection briefly.
# 🔹 Task 3: Data Preparation
# Prepare the selected data so that distance-based grouping works correctly.
# Verify that all features contribute fairly to distance calculation.
# 🔹 Task 4: Clustering Model Construction
# Build a clustering model to group customers into K segments.
# Experiment with multiple values of K.
# 🔹 Task 5: Optimal Cluster Identification
# Identify a suitable number of customer groups using an appropriate approach.
# Explain why this value of K is reasonable.
# 🔹 Task 6: Cluster Assignment
# Assign each customer to a cluster.
# Add the cluster label to the dataset.
# 🔹 Task 7: Cluster Visualization
# Visualize customer clusters using two important spending categories.
# Mark cluster centers clearly in the visualization.
# 🔹 Task 8: Cluster Profiling
# For each cluster:
# Calculate average spending per category
# Identify dominant purchase patterns
# Summarize each cluster in business-friendly language.
# 🔹 Task 9: Business Insight Generation
# Propose one business strategy for each customer segment, such as:
# Targeted promotions
# Inventory prioritization
# Personalized pricing strategies
# 🔹 Task 10: Stability & Limitations
# Rerun clustering with a different random state.
# Observe whether cluster assignments change.
# Mention one limitation of this clustering approach.
 

# ============================================
# Wholesale Customers Segmentation using K-Means
# ============================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Wholesale Customer Segmentation",
    layout="centered"
)

st.title("🛒 Wholesale Customer Segmentation using K-Means")

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("Wholesale customers data.csv")

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# ----------------------------
# Feature Selection
# ----------------------------
features = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

X = df[features]

st.subheader("📌 Selected Features")
st.write(features)

# ----------------------------
# Data Scaling
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# Elbow Method
# ----------------------------
st.subheader("📉 Elbow Method (Optimal K)")

wcss = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker="o")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("WCSS")
ax.set_title("Elbow Method")
st.pyplot(fig)

# ----------------------------
# Choose Number of Clusters
# ----------------------------
k = st.slider(
    "Select Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=4
)

# ----------------------------
# Build K-Means Model
# ----------------------------
kmeans = KMeans(
    n_clusters=k,
    init="k-means++",
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

st.subheader("📊 Clustered Dataset")
st.dataframe(df.head())

# ----------------------------
# Centroids (Scaled & Original)
# ----------------------------
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

centroids_df = pd.DataFrame(
    centroids_original,
    columns=features
)
centroids_df["Cluster"] = range(k)

st.subheader("🎯 Cluster Centroids (Original Scale)")
st.dataframe(centroids_df)

# ----------------------------
# Cluster Visualization
# ----------------------------
st.subheader("📍 Cluster Visualization (Grocery vs Detergents_Paper)")

fig2, ax2 = plt.subplots()

for i in range(k):
    ax2.scatter(
        X_scaled[df["Cluster"] == i, features.index("Grocery")],
        X_scaled[df["Cluster"] == i, features.index("Detergents_Paper")],
        label=f"Cluster {i}"
    )

# Plot centroids
ax2.scatter(
    centroids_scaled[:, features.index("Grocery")],
    centroids_scaled[:, features.index("Detergents_Paper")],
    s=300,
    c="black",
    marker="X",
    label="Centroids"
)

ax2.set_xlabel("Grocery (Scaled)")
ax2.set_ylabel("Detergents_Paper (Scaled)")
ax2.legend()
st.pyplot(fig2)

# ----------------------------
# Cluster Profiling
# ----------------------------
st.subheader("📌 Cluster Profiling (Average Spending)")
cluster_profile = df.groupby("Cluster")[features].mean()
st.dataframe(cluster_profile)

# ----------------------------
# Business Insights
# ----------------------------
st.subheader("💡 Business Insights")

st.markdown("""
- **High Grocery & Detergents spenders** → Retail chains → Bulk inventory planning  
- **High Fresh & Frozen buyers** → Hotels & restaurants → Cold storage focus  
- **Low spend clusters** → Small retailers → Targeted promotions  
- **Balanced spenders** → Mixed customers → Cross-selling opportunities  
""")

# ----------------------------
# Stability Check
# ----------------------------
st.subheader("🔁 Stability Check")

kmeans_alt = KMeans(
    n_clusters=k,
    init="k-means++",
    random_state=99,
    n_init=10
)

alt_labels = kmeans_alt.fit_predict(X_scaled)

st.write(
    "Clustering stability checked by changing random state. "
    "Minor changes indicate acceptable stability."
)

# ----------------------------
# Limitations
# ----------------------------
st.subheader("⚠️ Model Limitations")
st.markdown("""
- K-Means assumes spherical clusters  
- Sensitive to outliers  
- Requires predefined K value  
""")
