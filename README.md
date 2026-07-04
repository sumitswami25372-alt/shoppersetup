# Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce

##  Project Overview

Shopper Spectrum is an end-to-end Machine Learning and Data Analytics project that analyzes customer transaction data from an online retail business.

The project performs:

* Customer Segmentation using RFM Analysis and K-Means Clustering
* Product Recommendation using Item-Based Collaborative Filtering
* Exploratory Data Analysis (EDA)
* Interactive Streamlit Dashboard

The goal is to help businesses understand customer behavior, improve marketing strategies, and provide personalized product recommendations.

---

## Objectives

* Analyze customer purchasing behavior
* Perform data preprocessing and feature engineering
* Build customer segments using clustering techniques
* Generate personalized product recommendations
* Visualize business insights through interactive dashboards

---

## Dataset Description
 Column      | Description                  |
 ----------- | ---------------------------- |
 InvoiceNo   | Transaction Number           |
 StockCode   | Product Code                 |
 Description | Product Name                 |
 Quantity    | Number of Products Purchased |
 InvoiceDate | Transaction Date             |
 UnitPrice   | Product Price                |
 CustomerID  | Unique Customer Identifier   |
 Country     | Customer Country             |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Plotly
* Streamlit
* Joblib

---

## Project Workflow

### 1. Data Collection & Understanding

* Dataset Exploration
* Missing Value Analysis
* Duplicate Record Detection

### 2. Data Preprocessing

* Remove Missing Customer IDs
* Remove Cancelled Invoices
* Remove Invalid Quantities
* Remove Invalid Prices

### 3. Exploratory Data Analysis

* Country-wise Transaction Analysis
* Top Selling Products
* Purchase Trends
* Revenue Distribution
* RFM Analysis
* Correlation Analysis

### 4. Customer Segmentation

* RFM Feature Engineering
* Feature Scaling
* Elbow Method
* Silhouette Analysis
* K-Means Clustering

Customer Segments:

* High-Value Customers
* Regular Customers
* Occasional Customers
* At-Risk Customers

### 5. Product Recommendation System

* Customer-Product Matrix Creation
* Item-Based Collaborative Filtering
* Cosine Similarity Calculation
* Top Product Recommendations

### 6. Streamlit Dashboard

Features:

* Dataset Overview
* EDA Visualizations
* Customer Segmentation Analysis
* Product Recommendation System
* Business Insights Dashboard

---

## Project Structure

shopper-spectrum/

├── app.py

├── requirements.txt

├── README.md

├── data/

│ └── OnlineRetail.csv

├── notebooks/

│ └── Shopper_Spectrum.ipynb

├── models/

│ ├── kmeans_model.pkl

│ ├── rfm_scaler.pkl

│ └── product_similarity.pkl

├── outputs/

│ ├── customer_segments.csv

│ └── visualizations/

└── screenshots/

---

## Installation

Clone the repository:

git clone https://github.com/yourusername/shopper-spectrum.git

Move to project directory:

cd shopper-spectrum

Install dependencies:

pip install -r requirements.txt

Run Streamlit application:

streamlit run app.py

---

## Machine Learning Models

### Customer Segmentation

Algorithm:

* K-Means Clustering

Evaluation:

* Elbow Method
* Silhouette Score

### Recommendation System

Algorithm:

* Item-Based Collaborative Filtering

Similarity Measure:

* Cosine Similarity

---

## Results

* Successfully segmented customers into meaningful groups.
* Identified High-Value and At-Risk customers.
* Built a recommendation engine capable of suggesting similar products.
* Generated actionable business insights for marketing and sales teams.

---

## Author

Sumit Swami

Data Analytics & Machine Learning Project

Labmentix Internship Project