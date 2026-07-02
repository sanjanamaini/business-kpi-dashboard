# Business KPI Dashboard — Online Retail

**Revenue, product, and customer-segment KPIs for a UK online retailer — analysis notebook plus a self-contained interactive dashboard.**

Built on the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail): 541,909 raw transactions (Dec 2010 – Dec 2011), cleaned to paid customer transactions.

## The dashboard

`dashboard.html` is a single self-contained file — open it in any browser, no server, no dependencies, no CDN. Interactive SVG charts (hover for exact values), automatic light/dark theme.

Regenerate it from the source data with:

```
pip install -r requirements.txt
python build_dashboard.py
```

## Headline numbers (computed from the cleaned data)

| | |
|---|---|
| Total revenue | **£8,911,407.90** |
| Orders | 18,532 |
| Customers | 4,338 |
| Average order value | £480.87 |

## The insight that matters most

RFM (recency / frequency / monetary) quartile segmentation shows how concentrated this business is:

| Segment | Customers | Revenue | Share of revenue |
|---|---|---|---|
| Champions | 609 (14%) | £4.58M | **51%** |
| Loyal | 914 (21%) | £2.02M | 23% |
| At risk | 646 (15%) | £1.04M | 12% |
| Hibernating | 1,504 (35%) | £0.77M | 9% |
| Recent / one-off | 665 (15%) | £0.50M | 6% |

**Half the revenue comes from 14% of customers.** The 646 "at risk" customers (high past frequency, gone quiet) hold £1.04M of demonstrated spend — that's the retention campaign with the clearest payback. Meanwhile the 1,504 hibernating customers contribute 9% of revenue; win-back spend there should be minimal.

## What's in the repo

```
main.ipynb           the analysis: cleaning → revenue KPIs → monthly trend →
                     top products/countries → RFM computation
build_dashboard.py   aggregates the data and emits dashboard.html
dashboard.html       the interactive dashboard (committed, ready to open)
Online Retail.xlsx   source data (UCI)
requirements.txt
```

## Tech stack

Python · pandas · openpyxl · matplotlib/seaborn (notebook) · hand-built inline SVG (dashboard — no chart library)
