# Lab 5: GCP Data Storage and Warehousing

## Additional Contributions Beyond Lab Requirements

For this lab, I went beyond the basic requirements by:
- **Selected a unique, real-world dataset**: Global Superstore Sales dataset instead of the standard Iris dataset
- **Implemented advanced SQL queries**: Created complex queries including profitability analysis, customer segmentation, and discount impact analysis
- **Built comprehensive Looker Studio dashboard**: Developed interactive visualizations with multiple chart types for deeper business insights

---

## Lab Setup

### 1. Create GCS Bucket

**Manual Setup (via GCP Console):**
1. Navigate to Google Cloud Console → Storage → Cloud Storage → Buckets
2. Click "Create Bucket"
3. Bucket name: `superstore-sales-data-sanskar`
4. Location: `us-east1` (Region)
5. Storage class: Standard
6. Click "Create"

**Command Line Setup:**

```bash
# Authenticate with Google Cloud
gcloud auth login

# Set your project
gcloud config set project lab-5-data-warehouse

# Verify bucket access
gsutil ls gs://superstore-sales-data-sanskar

# Create local directory for dataset
mkdir -p superstore-data

# Move downloaded dataset to working directory
mv ~/Downloads/archive/Global_Superstore2.csv superstore-data/

# Upload dataset to GCS bucket
gsutil cp superstore-data/Global_Superstore2.csv gs://superstore-sales-data-sanskar/data/Global_Superstore.csv

# Enable versioning on bucket
gsutil versioning set on gs://superstore-sales-data-sanskar

# Verify upload
gsutil ls gs://superstore-sales-data-sanskar/data/
```

---

### 2. Load Data into BigQuery

**Steps:**
1. Go to BigQuery Console → SQL workspace
2. Create Dataset:
   - Dataset ID: `superstore_analytics`
   - Location: `us-east1`
3. Create Table:
   - Source: Google Cloud Storage
   - File path: `gs://superstore-sales-data-sanskar/data/Global_Superstore.csv`
   - Format: CSV
   - Table name: `sales_data`
   - Schema: Auto-detect ✓

---

## SQL Queries Executed

### Query 1: View First 100 Rows
```sql
-- View first 100 rows
SELECT *
FROM `lab-5-data-warehouse.superstore_analytics.sales_data`
LIMIT 100;
```
**Purpose:** Initial data exploration and validation

---

### Query 2: Top 10 Most Profitable Products
```sql
-- Find top 10 most profitable products
SELECT 
    Product_Name,
    Category,
    Sub_Category,
    ROUND(SUM(Sales), 2) AS Total_Sales,
    ROUND(SUM(Profit), 2) AS Total_Profit,
    ROUND(SUM(Profit) / SUM(Sales) * 100, 2) AS Profit_Margin_Percent
FROM 
    `lab-5-data-warehouse.superstore_analytics.sales_data`
GROUP BY 
    Product_Name, Category, Sub_Category
ORDER BY 
    Total_Profit DESC
LIMIT 10;
```
**Insights:** Identified highest-margin products (Canon imageCLASS copiers showed 48%+ margins)

---

### Query 3: Top Customer Analysis
```sql
-- Identify and analyze top customers
SELECT 
    Customer_Name,
    Segment,
    Region,
    COUNT(DISTINCT Order_ID) AS Total_Orders,
    ROUND(SUM(Sales), 2) AS Lifetime_Value,
    ROUND(SUM(Profit), 2) AS Total_Profit_Generated,
    ROUND(AVG(Sales), 2) AS Avg_Order_Value,
    STRING_AGG(DISTINCT Category, ', ') AS Categories_Purchased
FROM 
    `lab-5-data-warehouse.superstore_analytics.sales_data`
GROUP BY 
    Customer_Name, Segment, Region
HAVING 
    SUM(Sales) > 10000
ORDER BY 
    Lifetime_Value DESC
LIMIT 20;
```
**Insights:** Corporate segment customers generate highest lifetime value ($25K+)

---

## Looker Studio Dashboard

### Setup
1. Go to [lookerstudio.google.com](https://lookerstudio.google.com)
2. Create Data Source → BigQuery → Connect to `superstore_analytics.sales_data`
3. Create Report

Visualizations are available in `looker_studio_report` folder

---

## Technologies Used
- Google Cloud Storage (data storage)
- BigQuery (data warehousing & SQL analytics)
- Looker Studio (data visualization)
- Global Superstore Dataset (51K+ sales records)