# Lab 5: GCP Data Storage and Warehousing

## Additional Contributions Beyond Lab Requirements

For this lab, I went beyond the basic requirements by:
- **Selected a unique, real-world dataset**: Global Superstore Sales dataset instead of the standard Iris dataset
- **Implemented advanced SQL queries**: Created complex queries including profitability analysis and customer segmentation
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

<img width="1495" height="613" alt="Screenshot 2025-11-16 at 1 12 23 PM" src="https://github.com/user-attachments/assets/7b3feda5-4c39-4294-999d-78a818e7c499" />


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
<img width="1512" height="402" alt="Screenshot 2025-11-16 at 1 20 16 PM" src="https://github.com/user-attachments/assets/f24b8f90-0adc-423c-a144-2c62f890db56" />
<img width="1242" height="517" alt="Screenshot 2025-11-16 at 1 21 32 PM" src="https://github.com/user-attachments/assets/8070b4b0-5b8c-4c57-9866-524cdbf61bfa" />


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

<img width="1493" height="757" alt="Screenshot 2025-11-16 at 1 25 44 PM" src="https://github.com/user-attachments/assets/e4352a48-c081-42b0-a89d-240273679ec1" />

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

<img width="1135" height="671" alt="Screenshot 2025-11-16 at 1 27 34 PM" src="https://github.com/user-attachments/assets/8a551814-bd23-486d-a50b-57c7f06e5eb1" />

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

<img width="1133" height="668" alt="Screenshot 2025-11-16 at 1 30 24 PM" src="https://github.com/user-attachments/assets/59df5922-a00e-4601-9894-4463633fa493" />


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

<img width="1134" height="667" alt="Screenshot 2025-11-16 at 1 32 15 PM" src="https://github.com/user-attachments/assets/f2f5d9bc-d272-4cbc-b478-8102da673d24" />

---

## Looker Studio Dashboard

### Setup
1. Go to [lookerstudio.google.com](https://lookerstudio.google.com)
2. Create Data Source → BigQuery → Connect to `superstore_analytics.sales_data`
3. Create Report

Visualization Report is available in `looker_studio_report` folder

<img width="1495" height="758" alt="Screenshot 2025-11-16 at 1 44 48 PM" src="https://github.com/user-attachments/assets/e2ac1caf-e285-4d57-be80-09d46cb1624f" />

---

## Technologies Used
- Google Cloud Storage (data storage)
- BigQuery (data warehousing & SQL analytics)
- Looker Studio (data visualization)
- Global Superstore Dataset (51K+ sales records)
