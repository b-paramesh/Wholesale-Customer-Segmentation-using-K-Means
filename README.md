# 🛒 Wholesale Customer Segmentation using K-Means

## 📌 Project Overview
This project applies **K-Means clustering** to the **Wholesale Customers Dataset** to segment customers based on their annual purchasing behavior.  
The goal is to help a wholesale distributor make **data-driven business decisions** related to inventory planning, marketing strategies, and customer targeting.

An **interactive Streamlit web application** is built to visualize clusters, centroids, and business insights.

---

## 🧠 Business Problem
A wholesale distributor serves multiple types of customers such as:
- Retail Stores  
- Cafés  
- Hotels  
- Restaurants  

Currently, all customers are treated the same, which leads to:
- Inefficient inventory management  
- Poor marketing effectiveness  
- Missed upselling opportunities  

👉 **Objective:** Group customers based on purchasing patterns to enable targeted strategies.

---

## 📊 Dataset Information
- **Dataset Name:** Wholesale Customers Dataset  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set  

### Features Used
| Feature | Description |
|------|------------|
| Fresh | Annual spending on fresh products |
| Milk | Annual spending on milk products |
| Grocery | Annual spending on grocery items |
| Frozen | Annual spending on frozen products |
| Detergents_Paper | Annual spending on detergents & paper |
| Delicassen | Annual spending on delicatessen items |

> Columns like `Channel` and `Region` were excluded as they do not directly represent spending behavior.

---

## 🛠️ Technologies Used
- **Python**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **Streamlit**
- **Git & GitHub**

---

## 🔍 Project Workflow
1. Data loading and exploration  
2. Feature selection for customer behavior  
3. Feature scaling using `StandardScaler`  
4. Optimal cluster selection using the **Elbow Method**  
5. K-Means clustering model training  
6. Cluster assignment and labeling  
7. Cluster visualization with centroids  
8. Cluster profiling and business interpretation  
9. Stability check using different random states  
10. Identification of model limitations  

---

## 📉 Elbow Method
The Elbow Method is used to identify an optimal number of clusters by analyzing **Within-Cluster Sum of Squares (WCSS)**.

---

## 🎯 Cluster Centroids
- Actual K-Means centroids are extracted from the model
- Centroids are inverse-transformed to original scale for business interpretation
- Centroids are visualized on cluster plots

---

## 📈 Visualizations
- Interactive elbow plot  
- Cluster visualization using:
  - Grocery vs Detergents_Paper
- Centroids clearly marked on the plot

---

## 💡 Business Insights
- **High Grocery & Detergents buyers:** Likely retail chains → Bulk inventory focus  
- **High Fresh & Frozen buyers:** HoReCa customers → Cold storage optimization  
- **Low spending clusters:** Small customers → Targeted promotions  
- **Balanced spenders:** Mixed segment → Cross-selling opportunities  

---

## ⚠️ Model Limitations
- Requires predefining the number of clusters (K)
- Sensitive to outliers
- Assumes spherical cluster shapes

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/b-paramesh/Wholesale-Customer-Segmentation-using-K-Means.git
cd Wholesale-Customer-Segmentation-using-K-Means
2️⃣ Create and activate virtual environment
python -m venv venv
venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Run Streamlit app
streamlit run kmeans_Clustering_wholesale_customers.py
🌐 Streamlit App
The application provides:

Interactive cluster selection

Real-time visualization

Business-friendly summaries

📌 Future Enhancements
PCA-based 2D & 3D clustering

Customer segment prediction

Download clustered dataset

Cloud deployment (AWS / Azure)

👨‍💻 Author
Paramesh B
Machine Learning & Data Science Enthusiast

🔗 GitHub: https://github.com/b-paramesh

⭐ If you find this project useful
Give it a ⭐ on GitHub!


---

## ✅ How to add this to your repo

```bash
notepad README.md
Paste the content → Save → then:

git add README.md
git commit -m "Add project README"
git push
