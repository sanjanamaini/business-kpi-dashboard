"""UCI Online Retail: cleaning with a ledger, cancellation netting, RFM, and probabilistic customer value.

Cleaning decisions (each one's effect is recorded in the ledger):
- rows without a CustomerID cannot be attributed to a customer and are set aside;
- exact duplicate rows are removed;
- lines with a unit price of zero or less are free items or corrections and carry no revenue;
- postage, carriage, fees, bank charges, manual adjustments, samples and discounts are not
  merchandise and are kept apart from product revenue;
- every cancellation line is netted against the same customer's most recent earlier purchase of the
  same stock code, so a cancelled order no longer counts as a sale. Cancellations with no matching
  purchase in the window (bought before December 2010) stay as negative adjustments.

Models (Fader, Hardie and Lee 2005, "Counting your customers the easy way" and "RFM and CLV"):
- BG/NBD for the number of future purchase days, fitted by maximum likelihood;
- Gamma-Gamma for the average value of a purchase day.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

NON_MERCH = {"POST", "DOT", "C2", "M", "m", "BANK CHARGES", "AMAZONFEE", "CRUK", "D", "S", "B", "PADS"}


def load(xlsx: Path) -> pd.DataFrame:
    cache = xlsx.with_suffix(".parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_excel(xlsx, dtype={"InvoiceNo": str, "StockCode": str})
    df["Description"] = df["Description"].astype("string")
    df.to_parquet(cache, index=False)
    return df


def v1_clean(df: pd.DataFrame) -> pd.DataFrame:
    """The cleaning the published numbers used (build_dashboard.py before v2)."""
    d = df.dropna(subset=["CustomerID"])
    d = d[(d["Quantity"] > 0) & (d["UnitPrice"] > 0)].copy()
    d["TotalPrice"] = d["Quantity"] * d["UnitPrice"]
    return d


def net_cancellations(purchases: pd.DataFrame, cancels: pd.DataFrame) -> tuple:
    """Subtract each cancelled quantity from earlier purchase lines of the same customer and stock
    code: first lines at the same unit price (most recent first), then any price (most recent first).
    Matching on price first matters: one customer cancelled 60 baskets at £4.95 minutes after a
    mis-keyed line of 60 at £649.50, and plain last-in-first-out would cancel the wrong one.
    Returns purchases with a remaining quantity, and the unmatched part of each cancellation
    (quantity that had no earlier purchase in the data)."""
    p = purchases.sort_values("InvoiceDate").copy()
    p["remaining"] = p["Quantity"].astype(float)
    book = defaultdict(list)
    for idx, cust, code in zip(p.index, p["CustomerID"], p["StockCode"]):
        book[(cust, code)].append(idx)
    unmatched = []
    remaining = p["remaining"].to_dict()
    when = p["InvoiceDate"].to_dict()
    price = p["UnitPrice"].to_dict()
    for row in cancels.sort_values("InvoiceDate").itertuples():
        need = -row.Quantity
        cands = [i for i in reversed(book.get((row.CustomerID, row.StockCode), [])) if when[i] <= row.InvoiceDate]
        ordered = [i for i in cands if abs(price[i] - row.UnitPrice) < 1e-9] + [i for i in cands if abs(price[i] - row.UnitPrice) >= 1e-9]
        for idx in ordered:
            if need <= 0:
                break
            if remaining[idx] <= 0:
                continue
            take = min(need, remaining[idx])
            remaining[idx] -= take
            need -= take
        if need > 0:
            unmatched.append({"CustomerID": row.CustomerID, "InvoiceDate": row.InvoiceDate, "StockCode": row.StockCode,
                              "Quantity": -need, "UnitPrice": row.UnitPrice})
    p["remaining"] = pd.Series(remaining)
    p["Value"] = p["remaining"] * p["UnitPrice"]
    return p, pd.DataFrame(unmatched)


def ledger_clean(df: pd.DataFrame) -> tuple:
    """Clean step by step; return (merchandise purchase lines after netting, unmatched cancellations,
    service lines, ledger)."""
    rows = []
    d = df.copy()
    d["Line"] = d["Quantity"] * d["UnitPrice"]

    def note(step, data, removed_rows=0, removed_value=0.0):
        rows.append({"step": step, "rows": len(data), "rows_removed": removed_rows, "value_removed": removed_value,
                     "gross_sales": float(data.loc[data["Line"] > 0, "Line"].sum())})

    note("raw file", d)
    gone = d[d["CustomerID"].isna()]
    d = d[d["CustomerID"].notna()]
    note("drop rows without a customer ID", d, len(gone), float(gone["Line"].sum()))
    dup = d.duplicated()
    note_val = float(d.loc[dup, "Line"].sum())
    d = d[~dup]
    note("drop exact duplicate rows", d, int(dup.sum()), note_val)
    free = d["UnitPrice"] <= 0
    d = d[~free]
    note("drop zero-price lines", d, int(free.sum()), 0.0)
    svc = d["StockCode"].isin(NON_MERCH)
    service = d[svc].copy()
    d = d[~svc]
    note("set aside postage, fees and adjustments", d, int(svc.sum()), float(service["Line"].sum()))
    is_cancel = d["InvoiceNo"].str.startswith("C") | (d["Quantity"] < 0)
    purchases, cancels = d[~is_cancel], d[is_cancel]
    netted, unmatched = net_cancellations(purchases, cancels)
    cancelled_value = float((netted["Quantity"] - netted["remaining"]).mul(netted["UnitPrice"]).sum())
    rows.append({"step": "net cancellations against the purchases they reverse", "rows": int((netted["remaining"] > 0).sum()),
                 "rows_removed": int(len(cancels)), "value_removed": -cancelled_value,
                 "gross_sales": float(netted["Value"].sum())})
    # Manual adjustment lines ("M") that exactly reverse a purchase line's value on the same day are
    # corrections of mis-keyed lines (e.g. 60 x £649.50 entered for 60 x £4.95); net them too.
    manual = service[service["StockCode"].isin({"M", "m"}) & (service["Line"] < 0)]
    fixed, fixed_value = 0, 0.0
    for row in manual.itertuples():
        same_day = netted[(netted["CustomerID"] == row.CustomerID) & (netted["InvoiceDate"].dt.normalize() == row.InvoiceDate.normalize())
                          & ((netted["Value"] + row.Line).abs() < 0.01) & (netted["remaining"] > 0)]
        if len(same_day):
            i = same_day.index[0]
            fixed_value += float(netted.at[i, "Value"])
            netted.at[i, "remaining"], netted.at[i, "Value"] = 0.0, 0.0
            fixed += 1
    rows.append({"step": "net manual corrections that exactly reverse a same-day line", "rows": int((netted["remaining"] > 0).sum()),
                 "rows_removed": fixed, "value_removed": -fixed_value, "gross_sales": float(netted["Value"].sum())})
    netted = netted[netted["remaining"] > 0].copy()
    return netted, unmatched, service, pd.DataFrame(rows)


def rfm_table(lines: pd.DataFrame, ref_date: pd.Timestamp, value_col: str, adjustments: pd.Series = None) -> pd.DataFrame:
    rfm = lines.groupby("CustomerID").agg(Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
                                          Frequency=("InvoiceNo", "nunique"), Monetary=(value_col, "sum"))
    if adjustments is not None:
        rfm["Monetary"] = rfm["Monetary"].add(adjustments, fill_value=0).reindex(rfm.index)
    return rfm


def rfm_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """v1's quartile scoring and labelling rules, unchanged, so segments can be compared."""
    r = rfm.copy()
    r["R"] = pd.qcut(r["Recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    r["F"] = pd.qcut(r["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    r["M"] = pd.qcut(r["Monetary"], 4, labels=[1, 2, 3, 4]).astype(int)
    r["Segment"] = np.select(
        [(r.R >= 4) & (r.F >= 4), (r.F >= 3) & (r.R >= 3), (r.R >= 3) & (r.F <= 2), (r.R <= 2) & (r.F >= 3)],
        ["Champions", "Loyal", "Recent / one-off", "At risk"], default="Hibernating")
    return r


def daily_transactions(lines: pd.DataFrame, value_col: str = "Value") -> pd.DataFrame:
    """One row per customer per day with positive merchandise value: a 'purchase day'."""
    t = lines.assign(day=lines["InvoiceDate"].dt.normalize()).groupby(["CustomerID", "day"])[value_col].sum()
    return t[t > 0].rename("value").reset_index()


def summarise(tx: pd.DataFrame, end: pd.Timestamp, unit_days: float = 7.0) -> pd.DataFrame:
    """Per customer: x = repeat purchase days, t_x = time of last purchase since the first, T = age at
    `end` (all in weeks), and m = mean value of the repeat purchase days (Gamma-Gamma convention)."""
    t = tx[tx["day"] <= end]
    g = t.groupby("CustomerID")
    first, last = g["day"].min(), g["day"].max()
    s = pd.DataFrame({"x": g.size() - 1, "t_x": (last - first).dt.days / unit_days, "T": (end - first).dt.days / unit_days})
    rep = t.merge(first.rename("first"), left_on="CustomerID", right_index=True)
    rep = rep[rep["day"] > rep["first"]]
    s["m"] = rep.groupby("CustomerID")["value"].mean()
    s["first_value"] = t.sort_values("day").groupby("CustomerID")["value"].first()
    return s


# --- BG/NBD -------------------------------------------------------------------------------
def bgnbd_negll(log_params, x, tx, T):
    r, alpha, a, b = np.exp(log_params)
    a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    a3 = -(r + x) * np.log(alpha + T)
    with np.errstate(divide="ignore", invalid="ignore"):
        a4 = np.where(x > 0, np.log(a) - np.log(np.maximum(b + x - 1, 1e-12)) - (r + x) * np.log(alpha + tx), -np.inf)
    return -np.sum(a1 + a2 + np.logaddexp(a3, a4))


def fit_bgnbd(s: pd.DataFrame) -> dict:
    x, tx, T = s["x"].to_numpy(float), s["t_x"].to_numpy(float), s["T"].to_numpy(float)
    best = None
    for start in ([0.0, 1.0, 0.0, 0.0], [-0.5, 2.0, -1.0, 1.0], [0.5, 3.0, 0.5, 2.0]):
        res = minimize(bgnbd_negll, np.array(start), args=(x, tx, T), method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    r, alpha, a, b = np.exp(best.x)
    return {"r": r, "alpha": alpha, "a": a, "b": b, "negll": float(best.fun), "converged": bool(best.success)}


def p_alive(p: dict, x, tx, T):
    r, alpha, a, b = p["r"], p["alpha"], p["a"], p["b"]
    x, tx, T = (np.asarray(v, float) for v in (x, tx, T))
    odds = np.where(x > 0, a / (b + x - 1) * ((alpha + T) / (alpha + tx)) ** (r + x), 0.0)
    return 1.0 / (1.0 + odds)


def expected_purchases(p: dict, t: float, x, tx, T):
    """E[number of purchase days in (T, T + t] | x, t_x, T]."""
    r, alpha, a, b = p["r"], p["alpha"], p["a"], p["b"]
    x, tx, T = (np.asarray(v, float) for v in (x, tx, T))
    z = t / (alpha + T + t)
    hyp = hyp2f1(r + x, b + x, a + b + x - 1, z)
    num = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp)
    den = 1 + np.where(x > 0, a / (b + x - 1) * ((alpha + T) / (alpha + tx)) ** (r + x), 0.0)
    return num / den


def nbd_negll(log_params, x, T):
    """BG/NBD with no dropout (b -> infinity): Poisson purchasing with gamma-distributed rates."""
    r, alpha = np.exp(log_params)
    return -np.sum(gammaln(r + x) - gammaln(r) + r * np.log(alpha) - (r + x) * np.log(alpha + T))


def fit_nbd(s: pd.DataFrame) -> dict:
    x, T = s["x"].to_numpy(float), s["T"].to_numpy(float)
    res = minimize(nbd_negll, np.array([0.0, 1.0]), args=(x, T), method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-9})
    r, alpha = np.exp(res.x)
    return {"r": r, "alpha": alpha, "negll": float(res.fun), "converged": bool(res.success)}


def nbd_expected(p: dict, t: float, x, T):
    """E[purchase days in (T, T + t] | x, T] = t (r + x) / (alpha + T)."""
    return t * (p["r"] + np.asarray(x, float)) / (p["alpha"] + np.asarray(T, float))


def features_at(tx: pd.DataFrame, lines: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Customer features using only data on or before `origin` (for the machine-learning model)."""
    s = summarise(tx, origin)
    t = tx[tx["day"] <= origin]
    li = lines[lines["InvoiceDate"] <= origin + pd.Timedelta(days=1)]
    s["gap"] = s["T"] - s["t_x"]
    s["spend"] = t.groupby("CustomerID")["value"].sum()
    s["mean_value"] = t.groupby("CustomerID")["value"].mean()
    s["max_value"] = t.groupby("CustomerID")["value"].max()
    s["products"] = li.groupby("CustomerID")["StockCode"].nunique()
    s["uk"] = (li.groupby("CustomerID")["Country"].agg(lambda c: c.mode().iat[0]) == "United Kingdom").astype(int)
    s["last90_spend"] = t[t["day"] > origin - pd.Timedelta(days=90)].groupby("CustomerID")["value"].sum()
    s["last90_spend"] = s["last90_spend"].fillna(0)
    return s


# --- Gamma-Gamma --------------------------------------------------------------------------
def gg_negll(log_params, x, m):
    p, q, g = np.exp(log_params)
    ll = (gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(g) + (p * x - 1) * np.log(m)
          + p * x * np.log(x) - (p * x + q) * np.log(g + m * x))
    return -np.sum(ll)


def fit_gg(x, m) -> dict:
    x, m = np.asarray(x, float), np.asarray(m, float)
    best = None
    for start in ([1.0, 1.0, 5.0], [0.0, 2.0, 3.0], [1.5, 0.5, 6.0]):
        res = minimize(gg_negll, np.array(start), args=(x, m), method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    p, q, g = np.exp(best.x)
    return {"p": p, "q": q, "gamma": g, "negll": float(best.fun), "converged": bool(best.success)}


def expected_value(gp: dict, x, m, fallback: float):
    """E[mean purchase value | x repeat purchases averaging m]; customers with no repeat purchase get
    the population mean (p * gamma / (q - 1))."""
    p, q, g = gp["p"], gp["q"], gp["gamma"]
    x, m = np.asarray(x, float), np.asarray(m, float)
    pop = p * g / (q - 1)
    with np.errstate(invalid="ignore"):
        cond = (g + m * x) * p / (p * x + q - 1)
    return np.where((x > 0) & np.isfinite(m), cond, pop if np.isfinite(pop) else fallback)


def lorenz(values: np.ndarray) -> tuple:
    v = np.sort(np.asarray(values, float))
    cum = np.concatenate([[0], np.cumsum(v)]) / v.sum()
    pop = np.linspace(0, 1, len(cum))
    gini = 1 - 2 * np.trapz(cum, pop)
    return pop, cum, float(gini)


def capture(actual: np.ndarray, score: np.ndarray, frac: float) -> float:
    """Share of the actual total held by the top `frac` of customers ranked by `score`."""
    order = np.argsort(-np.asarray(score, float), kind="stable")
    n = int(np.ceil(frac * len(order)))
    a = np.asarray(actual, float)
    return float(a[order[:n]].sum() / a.sum())
