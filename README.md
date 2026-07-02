# Business KPI Dashboard (Online Retail Analysis)

An exploratory data analysis notebook on the classic UCI "Online Retail" transaction dataset (`Online Retail.xlsx`, 541,909 rows of UK e-commerce invoices, Dec 2010–Dec 2011). Despite the repo name, this is a single Jupyter notebook doing revenue/RFM analysis — there is no dashboard application (no Power BI/Tableau file, no web app) in this repo.

## Tech Stack

- Python, pandas
- matplotlib, seaborn
- openpyxl (for reading the `.xlsx` source file)

## How It Works

1. **Load & clean**: reads `Online Retail.xlsx`, removes cancelled orders (`InvoiceNo` starting with "C"), removes non-positive quantities, and drops rows with missing `CustomerID` (541,909 → 397,924 rows).
2. **Revenue calculation**: adds a `TotalPrice = Quantity × UnitPrice` column; total revenue across the cleaned dataset computes to £8,911,407.90.
3. **Time-series and breakdown charts**: monthly revenue trend, top 10 products by revenue, and top 10 countries by revenue.
4. **RFM analysis**: computes Recency, Frequency, and Monetary value per customer (`CustomerID`), using one day after the last invoice date (2011-12-10) as the reference date, then charts the top 10 customers by spend, by purchase frequency, and by recency.

## Run

Requires `Online Retail.xlsx` in the same directory (included in this repo). Open `main.ipynb` and run all cells; needs `pandas`, `matplotlib`, `seaborn`, `openpyxl` (the notebook itself contains `pip install` cells for these).
