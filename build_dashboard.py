"""Build a self-contained interactive KPI dashboard (dashboard.html) from the
Online Retail dataset. Aggregates are computed here; the HTML renders them as
inline SVG with no external dependencies, so the file works offline and on
GitHub Pages as-is.

Run:  pip install pandas openpyxl && python build_dashboard.py
"""
import json

import pandas as pd

SRC = "Online Retail.xlsx"
OUT = "dashboard.html"


def load_and_clean() -> pd.DataFrame:
    df = pd.read_excel(SRC)
    df = df.dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def rfm_segments(df: pd.DataFrame) -> pd.DataFrame:
    ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    # Quartile scores: R is better when low, F/M better when high.
    rfm["R"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4]).astype(int)

    def label(row):
        if row.R >= 4 and row.F >= 4:
            return "Champions"
        if row.F >= 3 and row.R >= 3:
            return "Loyal"
        if row.R >= 3 and row.F <= 2:
            return "Recent / one-off"
        if row.R <= 2 and row.F >= 3:
            return "At risk"
        return "Hibernating"

    rfm["Segment"] = rfm.apply(label, axis=1)
    return rfm


def build_payload(df: pd.DataFrame) -> dict:
    monthly = df.groupby(df["InvoiceDate"].dt.to_period("M"))["TotalPrice"].sum()
    top_products = df.groupby("Description")["TotalPrice"].sum().nlargest(10)
    top_countries = df.groupby("Country")["TotalPrice"].sum().nlargest(10)
    rfm = rfm_segments(df)
    seg = rfm.groupby("Segment").agg(customers=("Segment", "size"), revenue=("Monetary", "sum"))
    seg_order = ["Champions", "Loyal", "Recent / one-off", "At risk", "Hibernating"]
    seg = seg.reindex([s for s in seg_order if s in seg.index])

    return {
        "kpis": {
            "revenue": round(float(df["TotalPrice"].sum()), 2),
            "orders": int(df["InvoiceNo"].nunique()),
            "customers": int(df["CustomerID"].nunique()),
            "aov": round(float(df.groupby("InvoiceNo")["TotalPrice"].sum().mean()), 2),
            "date_min": str(df["InvoiceDate"].min().date()),
            "date_max": str(df["InvoiceDate"].max().date()),
        },
        "monthly": [{"m": str(k), "v": round(float(v), 2)} for k, v in monthly.items()],
        "products": [{"name": k[:38], "v": round(float(v), 2)} for k, v in top_products.items()],
        "countries": [{"name": k, "v": round(float(v), 2)} for k, v in top_countries.items()],
        "segments": [
            {
                "name": k,
                "customers": int(r.customers),
                "revenue": round(float(r.revenue), 2),
            }
            for k, r in seg.iterrows()
        ],
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Online Retail — KPI Dashboard</title>
<style>
:root {
  --surface: #fcfcfb; --page: #f9f9f7;
  --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,.10);
  --seq: #2a78d6; --seq-deep: #1c5cab;
  --s1: #2a78d6; --s2: #1baf7a; --s3: #eda100; --s4: #008300; --s5: #4a3aa7;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --page: #0d0d0d;
    --ink: #fff; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
    --seq: #3987e5; --seq-deep: #6da7ec;
    --s1: #3987e5; --s2: #199e70; --s3: #c98500; --s4: #008300; --s5: #9085e9;
  }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 28px 20px 60px; }
h1 { font-size: 22px; margin: 0 0 2px; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 22px; }
.tiles { display:grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:16px; }
.tile { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px 16px; }
.tile .v { font-size: 24px; font-weight: 700; }
.tile .l { font-size: 12px; color: var(--muted); margin-top:2px; }
.grid2 { display:grid; grid-template-columns: repeat(auto-fit,minmax(420px,1fr)); gap:12px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px 18px; }
.card h2 { font-size:14px; font-weight:600; color:var(--ink2); margin:0 0 10px; }
.card.full { grid-column: 1 / -1; }
svg text { font: 11px system-ui, sans-serif; fill: var(--muted); }
svg .val { fill: var(--ink2); font-weight: 600; }
.bar { fill: var(--seq); rx: 3; }
.bar:hover { fill: var(--seq-deep); }
#tip { position:fixed; pointer-events:none; background:var(--surface); color:var(--ink);
  border:1px solid var(--border); border-radius:8px; padding:6px 10px; font-size:12px;
  box-shadow:0 2px 8px rgba(0,0,0,.25); opacity:0; transition:opacity .12s; z-index:5; }
.legend { display:flex; gap:16px; flex-wrap:wrap; font-size:12px; color:var(--ink2); margin-top:8px; }
.legend span::before { content:""; display:inline-block; width:9px; height:9px; border-radius:2px;
  margin-right:5px; background:var(--c); }
.note { color:var(--muted); font-size:12px; margin-top:8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Online Retail — KPI Dashboard</h1>
  <div class="sub" id="daterange"></div>
  <div class="tiles" id="tiles"></div>
  <div class="grid2">
    <div class="card full"><h2>Monthly revenue</h2><div id="monthly"></div></div>
    <div class="card"><h2>Top 10 products by revenue</h2><div id="products"></div></div>
    <div class="card"><h2>Top 10 countries by revenue</h2><div id="countries"></div></div>
    <div class="card full"><h2>Customer segments (RFM quartiles)</h2><div id="segments"></div>
      <div class="legend" id="seglegend"></div>
      <div class="note">Segments from recency/frequency/monetary quartile scores per customer.</div>
    </div>
  </div>
</div>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const fmtGBP = v => "£" + (v >= 1e6 ? (v/1e6).toFixed(2)+"M" : v >= 1e3 ? (v/1e3).toFixed(0)+"K" : v.toFixed(0));
const fmtN = v => v.toLocaleString("en-GB");
const tip = document.getElementById("tip");
function showTip(e, html) { tip.innerHTML = html; tip.style.opacity = 1;
  tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY - 10) + "px"; }
function hideTip() { tip.style.opacity = 0; }

// KPI tiles
const k = DATA.kpis;
document.getElementById("daterange").textContent =
  `UCI Online Retail dataset · ${k.date_min} to ${k.date_max} · cleaned transactions only`;
document.getElementById("tiles").innerHTML = [
  [fmtGBP(k.revenue), "Total revenue"],
  [fmtN(k.orders), "Orders"],
  [fmtN(k.customers), "Customers"],
  ["£" + k.aov.toFixed(2), "Avg order value"],
].map(([v,l]) => `<div class="tile"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");

// Monthly revenue line
(function () {
  const W = 1020, H = 240, P = {l:52, r:12, t:12, b:34};
  const d = DATA.monthly, max = Math.max(...d.map(x=>x.v));
  const x = i => P.l + i * (W-P.l-P.r) / (d.length-1);
  const y = v => P.t + (H-P.t-P.b) * (1 - v/max);
  let grid = "", labels = "";
  for (let g = 0; g <= 4; g++) {
    const v = max*g/4, yy = y(v);
    grid += `<line x1="${P.l}" y1="${yy}" x2="${W-P.r}" y2="${yy}" stroke="var(--grid)"/>`;
    labels += `<text x="${P.l-6}" y="${yy+4}" text-anchor="end">${fmtGBP(v)}</text>`;
  }
  const path = d.map((p,i) => (i?"L":"M") + x(i) + " " + y(p.v)).join(" ");
  const pts = d.map((p,i) =>
    `<circle cx="${x(i)}" cy="${y(p.v)}" r="8" fill="transparent" data-m="${p.m}" data-v="${p.v}"/>` +
    `<circle cx="${x(i)}" cy="${y(p.v)}" r="3" fill="var(--seq)" pointer-events="none"/>`).join("");
  const xticks = d.map((p,i) => i % 2 === 0 ?
    `<text x="${x(i)}" y="${H-10}" text-anchor="middle">${p.m.slice(2)}</text>` : "").join("");
  document.getElementById("monthly").innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" width="100%">${grid}${labels}
     <path d="${path}" fill="none" stroke="var(--seq)" stroke-width="2"/>${pts}${xticks}</svg>`;
  document.querySelectorAll("#monthly circle[data-m]").forEach(c => {
    c.addEventListener("mousemove", e => showTip(e, `<b>${c.dataset.m}</b><br>${fmtGBP(+c.dataset.v)}`));
    c.addEventListener("mouseleave", hideTip);
  });
})();

// Horizontal bar charts
function hbars(elId, rows, color) {
  const W = 480, rowH = 26, P = {l:8, r:70, t:4}, H = P.t + rows.length*rowH + 6;
  const max = Math.max(...rows.map(r=>r.v));
  let s = "";
  rows.forEach((r, i) => {
    const yy = P.t + i*rowH, bw = Math.max(4, (W-P.l-P.r) * r.v/max);
    s += `<rect class="bar" x="${P.l}" y="${yy+4}" width="${bw}" height="14" rx="3"
            ${color?`style="fill:${color}"`:""} data-n="${r.name.replace(/"/g,"&quot;")}" data-v="${r.v}"/>
          <text x="${P.l+6}" y="${yy+15}" style="fill:#fff" font-size="10">${r.name}</text>
          <text class="val" x="${P.l+bw+6}" y="${yy+15}">${fmtGBP(r.v)}</text>`;
  });
  document.getElementById(elId).innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%">${s}</svg>`;
  document.querySelectorAll(`#${elId} rect`).forEach(b => {
    b.addEventListener("mousemove", e => showTip(e, `<b>${b.dataset.n}</b><br>${fmtGBP(+b.dataset.v)}`));
    b.addEventListener("mouseleave", hideTip);
  });
}
hbars("products", DATA.products);
hbars("countries", DATA.countries);

// Segments: paired customer/revenue share bars
(function () {
  const segColors = ["var(--s1)","var(--s2)","var(--s3)","var(--s4)","var(--s5)"];
  const totC = DATA.segments.reduce((a,s)=>a+s.customers,0);
  const totR = DATA.segments.reduce((a,s)=>a+s.revenue,0);
  const W = 1020, rowH = 40, H = DATA.segments.length*rowH + 24;
  let s = "";
  DATA.segments.forEach((seg, i) => {
    const yy = i*rowH + 4, cPct = seg.customers/totC, rPct = seg.revenue/totR;
    const maxW = W - 320;
    s += `<text class="val" x="0" y="${yy+16}">${seg.name}</text>
      <rect x="150" y="${yy+4}" width="${Math.max(3,maxW*cPct)}" height="10" rx="3" fill="${segColors[i]}"
        data-t="<b>${seg.name}</b><br>${fmtN(seg.customers)} customers (${(cPct*100).toFixed(1)}%)"/>
      <text x="${156+maxW*cPct}" y="${yy+13}">${(cPct*100).toFixed(0)}% of customers</text>
      <rect x="150" y="${yy+18}" width="${Math.max(3,maxW*rPct)}" height="10" rx="3" fill="${segColors[i]}" opacity="0.45"
        data-t="<b>${seg.name}</b><br>${fmtGBP(seg.revenue)} revenue (${(rPct*100).toFixed(1)}%)"/>
      <text x="${156+maxW*rPct}" y="${yy+27}">${(rPct*100).toFixed(0)}% of revenue</text>`;
  });
  document.getElementById("segments").innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%">${s}</svg>`;
  document.getElementById("seglegend").innerHTML = DATA.segments.map((seg,i) =>
    `<span style="--c:${segColors[i]}">${seg.name} — ${fmtN(seg.customers)} customers, ${fmtGBP(seg.revenue)}</span>`).join("");
  document.querySelectorAll("#segments rect").forEach(b => {
    b.addEventListener("mousemove", e => showTip(e, b.dataset.t));
    b.addEventListener("mouseleave", hideTip);
  });
})();
</script>
</body>
</html>
"""


def main():
    df = load_and_clean()
    payload = build_payload(df)
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {OUT}")
    print(f"revenue £{payload['kpis']['revenue']:,.2f} | orders {payload['kpis']['orders']:,} | "
          f"customers {payload['kpis']['customers']:,} | AOV £{payload['kpis']['aov']}")
    for s in payload["segments"]:
        print(f"  {s['name']:<18} {s['customers']:>5} customers  £{s['revenue']:>12,.2f}")


if __name__ == "__main__":
    main()
