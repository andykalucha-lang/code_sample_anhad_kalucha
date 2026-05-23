"""
Diagnostic checks for panchayat-level financial data (2018-19).
Run from the directory containing your data files, or update the path constants below.

Requirements: pandas, numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Path constants ─────────────────────────────────────────────────────────────
DATA_DIR          = Path("path/to/your/data/directory")
MERGED_FILE       = DATA_DIR / "combined_data_2018-19.csv"
DISTRICT_GLOB     = "*_monthly_summary_2018-2019.csv"   # individual district files
OUTPUT_FILE       = DATA_DIR / "combined_data_2018-19_cleaned.csv"

FINANCIAL_YEAR    = "2018-2019"
EXPECTED_DISTRICTS = 75
# ──────────────────────────────────────────────────────────────────────────────

MONTHS = ["april", "may", "june", "july", "august", "september",
          "october", "november", "december", "january", "february", "march"]
PREFIXES  = MONTHS + ["total"]
ID_COLS   = ["financial_year", "district_name", "district_code",
             "block_name", "block_code", "panchayat_name", "panchayat_code"]
KEY_COLS  = ["district_code", "block_code", "panchayat_code"]
FIN_COLS  = ["total_payments", "total_payment_cancellation",
             "opening_balance", "total_receipts", "total_receipt_cancellation"]
METRICS   = ["utilization_index", "utilization_index_no_canc",
             "payment_ratio", "payment_ratio_no_canc"]

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(MERGED_FILE)
print(f"Loaded: {df.shape}")

files = sorted(DATA_DIR.glob(DISTRICT_GLOB))
print(f"Individual district files found: {len(files)}")

# ── Fix rotated columns (2018-19 specific: 4 financial cols shifted by 1) ─────
# Raw export has:  payments <- payment_cancellation <- ob_rejected <- receipt_cancellation <- payments
for p in PREFIXES:
    pay = df[f"{p}_payments"].copy()
    df[f"{p}_payments"]              = df[f"{p}_payment_cancellation"]
    df[f"{p}_payment_cancellation"]  = df[f"{p}_ob_rejected"]
    df[f"{p}_ob_rejected"]           = df[f"{p}_receipt_cancellation"]
    df[f"{p}_receipt_cancellation"]  = pay

print(f"Column rotation applied for {len(PREFIXES)} prefixes.")

# =============================================================================
# CHECK 1: District count
# =============================================================================
n = df["district_name"].nunique()
status = "PASS" if n == EXPECTED_DISTRICTS else "WARNING"
print(f"\nCheck 1 [{status}]: {n} unique districts (expected {EXPECTED_DISTRICTS})")
print(sorted(df["district_name"].unique()))

# =============================================================================
# CHECK 2: District name matching — merged vs source filenames
# =============================================================================
filename_districts = {f.name.replace("_monthly_summary_2018-2019.csv", "") for f in files}
merged_districts   = set(df["district_name"].unique())

in_files_only  = filename_districts - merged_districts
in_merged_only = merged_districts   - filename_districts

print(f"\nCheck 2:")
print(f"  In filenames but not merged : {in_files_only  or 'None'}")
print(f"  In merged but not filenames : {in_merged_only or 'None'}")
print("  PASS" if not in_files_only and not in_merged_only else "  FAIL: mismatch detected")

# =============================================================================
# CHECK 3: Cross-contamination — district_code maps to exactly one name
# =============================================================================
cross = df.groupby("district_code")["district_name"].nunique()
bad   = cross[cross > 1]
if len(bad) == 0:
    print("\nCheck 3 [PASS]: Each district_code maps to one district_name")
else:
    print(f"\nCheck 3 [FAIL]: {len(bad)} district_code(s) map to multiple names")
    for code in bad.index:
        print(f"  {code}: {df[df['district_code'] == code]['district_name'].unique()}")

# =============================================================================
# CHECK 4: Panchayat code global uniqueness
# =============================================================================
code_dist = df.groupby("panchayat_code")["district_name"].nunique()
reused    = code_dist[code_dist > 1]
if len(reused) == 0:
    print("\nCheck 4 [PASS]: All panchayat_codes globally unique")
else:
    print(f"\nCheck 4 [INFO]: {len(reused)} panchayat_code(s) appear in >1 district (expected if codes are within-district)")
    detail = (
        df[df["panchayat_code"].isin(reused.index)]
        .groupby("panchayat_code")["district_name"]
        .apply(lambda x: sorted(x.unique()))
        .reset_index()
    )
    detail["n"] = detail["district_name"].apply(len)
    print(detail.sort_values("n", ascending=False).to_string(index=False))

# =============================================================================
# CHECK 4b: Duplicate (district_code, block_code, panchayat_code) keys
# =============================================================================
dupes   = df.duplicated(subset=KEY_COLS, keep=False)
n_dupes = dupes.sum()
if n_dupes == 0:
    print("\nCheck 4b [PASS]: No duplicate composite keys")
else:
    print(f"\nCheck 4b [FAIL]: {n_dupes} duplicate rows")
    by_dist = (df[dupes].groupby("district_name").size()
               .reset_index(name="dup_rows").sort_values("dup_rows", ascending=False))
    print(by_dist.to_string(index=False))

# =============================================================================
# CHECK 4c: Drop exact duplicates; report any remaining key conflicts
# =============================================================================
before = len(df)
df     = df.drop_duplicates().reset_index(drop=True)
print(f"\nCheck 4c: Dropped {before - len(df)} exact duplicate rows ({before} -> {len(df)})")

remaining = df.duplicated(subset=KEY_COLS, keep=False).sum()
if remaining == 0:
    print("  PASS: No remaining key conflicts after dedup")
else:
    print(f"  WARNING: {remaining} rows share a composite key but differ in data (not exact duplicates)")
    dup_df = df[df.duplicated(subset=KEY_COLS, keep=False)].sort_values(KEY_COLS)
    print(f"  Districts affected:\n{dup_df.groupby('district_name').size().sort_values(ascending=False).to_string()}")

# =============================================================================
# CHECK 5: Remove district-total summary rows; check missing identifier values
# =============================================================================
for col in ["district_name", "panchayat_name", "block_name"]:
    mask = df[col].astype(str).str.contains("district total", case=False, na=False)
    if mask.sum() > 0:
        print(f"\nCheck 5: Removed {mask.sum()} 'district total' rows from '{col}'")
        df = df[~mask].reset_index(drop=True)

missing = df[ID_COLS].isnull().sum()
if missing.sum() == 0:
    print("\nCheck 5 [PASS]: No missing values in identifier columns")
else:
    print("\nCheck 5 [FAIL]: Missing values in identifier columns:")
    print(missing[missing > 0])

print(f"Shape after cleaning: {df.shape}")

# =============================================================================
# CHECK 6: Financial year consistency
# =============================================================================
fy = df["financial_year"].unique()
if len(fy) == 1 and fy[0] == FINANCIAL_YEAR:
    print(f"\nCheck 6 [PASS]: All rows have financial_year = '{FINANCIAL_YEAR}'")
else:
    print(f"\nCheck 6 [FAIL]: Unexpected financial_year values:\n{df['financial_year'].value_counts()}")

# =============================================================================
# CHECK 7: opening_balance — parse Indian comma format if needed
# =============================================================================
if df["opening_balance"].dtype == object:
    df["opening_balance"] = pd.to_numeric(
        df["opening_balance"].str.replace(",", "", regex=False), errors="coerce"
    )
    print(f"\nCheck 7: Parsed opening_balance from string — "
          f"min={df['opening_balance'].min():,.0f}, max={df['opening_balance'].max():,.0f}, "
          f"NaN={df['opening_balance'].isna().sum()}")
else:
    print(f"\nCheck 7 [PASS]: opening_balance already numeric")

# =============================================================================
# CHECK 8: Monthly totals arithmetic (sum of 12 months == reported total)
# =============================================================================
FIN_FIELDS = ["receipts", "payments", "receipt_cancellation", "payment_cancellation", "ob_rejected"]
print("\nCheck 8: Monthly-to-total arithmetic")
all_pass = True
for field in FIN_FIELDS:
    monthly_cols = [f"{m}_{field}" for m in MONTHS]
    total_col    = f"total_{field}"
    computed     = df[monthly_cols].sum(axis=1)
    reported     = df[total_col]
    bad          = ~np.isclose(computed, reported, atol=0.01, equal_nan=True)
    if bad.sum() == 0:
        print(f"  PASS: {total_col}")
    else:
        all_pass = False
        bad_rows = df[bad][["district_name", "panchayat_name", total_col]].copy()
        bad_rows["computed"] = computed[bad]
        bad_rows["diff"]     = bad_rows["computed"] - bad_rows[total_col]
        print(f"  FAIL: {total_col} — {bad.sum()} mismatched rows (first 5 shown):")
        print(bad_rows.head(5).to_string(index=False))

# =============================================================================
# CHECK 9: Negative values in financial columns
# =============================================================================
numeric_cols = df.select_dtypes(include="number").columns.difference(["opening_balance"])
neg_report   = {c: (df[c] < 0).sum() for c in numeric_cols if (df[c] < 0).sum() > 0}
if not neg_report:
    print("\nCheck 9 [PASS]: No negative values in financial columns")
else:
    print(f"\nCheck 9 [FAIL]: Negative values in {len(neg_report)} columns:")
    for col, cnt in neg_report.items():
        print(f"  {col}: {cnt}")

# =============================================================================
# CHECK 10: Panchayat row counts — merged vs individual files
# =============================================================================
merged_counts = df.groupby("district_name").size().to_dict()
file_counts, skipped = {}, {}
for f in files:
    name       = f.name.replace("_monthly_summary_2018-2019.csv", "")
    ind        = pd.read_csv(f, encoding="latin-1", engine="python", on_bad_lines="skip")
    file_counts[name] = len(ind)
    with open(f, encoding="latin-1") as fh:
        raw = sum(1 for _ in fh) - 1
    if raw != len(ind):
        skipped[name] = raw - len(ind)

if skipped:
    print("\nCheck 10 WARNING: Bad lines skipped:")
    for name, n in skipped.items():
        print(f"  {name}: {n}")

mismatches = [
    (d, file_counts.get(d, 0), merged_counts.get(d, 0))
    for d in sorted(set(list(merged_counts) + list(file_counts)))
    if file_counts.get(d, 0) != merged_counts.get(d, 0)
]
if not mismatches:
    print(f"\nCheck 10 [PASS]: Row counts match for all {len(merged_counts)} districts")
else:
    print(f"\nCheck 10 [FAIL]: Row count mismatches in {len(mismatches)} districts:")
    print(pd.DataFrame(mismatches, columns=["district", "file_rows", "merged_rows"]).to_string(index=False))

# =============================================================================
# CLEAN & COMPUTE: Parse financial columns + compute 4 metrics
# =============================================================================
for col in FIN_COLS:
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col].str.replace(",", "", regex=False), errors="coerce")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# utilization_index: net_payments / available_funds (with cancellations)
ui_num = df["total_payments"] - df["total_payment_cancellation"]
ui_den = df["opening_balance"] + df["total_receipts"] - df["total_receipt_cancellation"]
df["utilization_index"] = np.where(ui_den <= 0, np.nan, ui_num / ui_den)

# utilization_index_no_canc: gross payments / (OB + receipts)
ui_nc_den = df["opening_balance"] + df["total_receipts"]
df["utilization_index_no_canc"] = np.where(ui_nc_den <= 0, np.nan, df["total_payments"] / ui_nc_den)

# payment_ratio: net_payments / net_receipts (with cancellations)
pr_den = df["total_receipts"] - df["total_receipt_cancellation"]
df["payment_ratio"] = np.where(pr_den <= 0, np.nan, ui_num / pr_den)

# payment_ratio_no_canc: gross payments / gross receipts
df["payment_ratio_no_canc"] = np.where(
    df["total_receipts"] <= 0, np.nan, df["total_payments"] / df["total_receipts"]
)

print("\nMetrics computed:")
for m in METRICS:
    valid = df[m].notna().sum()
    print(f"  {m}: valid={valid}/{len(df)}, "
          f"min={df[m].min():.4f}, median={df[m].median():.4f}, max={df[m].max():.4f}")

# Dummy: UI > 1
df["ui_over1"] = np.where(df["utilization_index"].isna(), np.nan,
                          (df["utilization_index"] > 1).astype(float))
n_over1 = (df["ui_over1"] == 1).sum()
n_valid = df["ui_over1"].notna().sum()
print(f"\nui_over1: {n_over1}/{n_valid} ({n_over1/n_valid*100:.1f}%) have UI > 1")

# =============================================================================
# CHECK 12: District-level aggregate outliers (>3 SD from cross-district mean)
# =============================================================================
dist_agg = df.groupby("district_name").agg(
    total_payments_sum=("total_payments", "sum"),
    total_receipts_sum=("total_receipts", "sum"),
    n_panchayats=("panchayat_name", "size"),
).reset_index()

print("\nCheck 12: District-level aggregate outliers")
for col in ["total_payments_sum", "total_receipts_sum"]:
    mu, sd   = dist_agg[col].mean(), dist_agg[col].std()
    outliers = dist_agg[np.abs(dist_agg[col] - mu) > 3 * sd]
    if len(outliers) == 0:
        print(f"  PASS: No outliers for {col}")
    else:
        print(f"  WARNING: {len(outliers)} outlier district(s) for {col} (>3 SD, mean={mu:,.0f}, sd={sd:,.0f}):")
        print(outliers[["district_name", col, "n_panchayats"]].to_string(index=False))

# Diagnose top-2 outlier districts (generalised — no hardcoded names)
for agg_col in ["total_payments_sum", "total_receipts_sum"]:
    raw_col  = agg_col.replace("_sum", "")
    mu, sd   = dist_agg[agg_col].mean(), dist_agg[agg_col].std()
    outliers = dist_agg[dist_agg[agg_col] > mu + 3 * sd].sort_values(agg_col, ascending=False)
    for _, row in outliers.head(2).iterrows():
        name  = row["district_name"]
        sub   = df[df["district_name"] == name]
        m_d   = sub[raw_col].mean()
        s_d   = sub[raw_col].std()
        top10 = sub.nlargest(5, raw_col)[["block_name", "panchayat_name", "panchayat_code", raw_col]]
        print(f"\n  Diagnosing {name} ({agg_col}):")
        print(f"    District total: {sub[raw_col].sum():,.0f} | mean per panchayat: {m_d:,.0f}")
        extreme = sub[sub[raw_col] > m_d + 3 * s_d]
        print(f"    Panchayats >3 SD within district: {len(extreme)}")
        print(f"    Top 5 panchayats:\n{top10.to_string(index=False)}")

# =============================================================================
# CHECK 13: Zero-metric panchayats — breakdown by district
# =============================================================================
dist_total = df.groupby("district_name").size().reset_index(name="total_panchayats")
print("\nCheck 13: Zero-value panchayats per metric")
for metric in METRICS:
    zero_m = df[df[metric] == 0]
    print(f"\n  {metric}: {len(zero_m)} zero-value rows")
    if len(zero_m) > 0:
        zbd = (zero_m.groupby("district_name").size()
               .reset_index(name="zero_count")
               .merge(dist_total, on="district_name"))
        zbd["pct_zero"] = (zbd["zero_count"] / zbd["total_panchayats"] * 100).round(1)
        print(zbd.sort_values("pct_zero", ascending=False).head(10).to_string(index=False))
        high = zbd[zbd["pct_zero"] > 20]
        if len(high):
            print(f"  WARNING: {len(high)} district(s) have >20% zero-value panchayats")

# =============================================================================
# CHECK 14: NaN values — denominator breakdown
# =============================================================================
metric_denom = {
    "utilization_index":         ("opening_balance", "total_receipts", "total_receipt_cancellation"),
    "utilization_index_no_canc": ("opening_balance", "total_receipts", None),
    "payment_ratio":             (None,              "total_receipts", "total_receipt_cancellation"),
    "payment_ratio_no_canc":     (None,              "total_receipts", None),
}

print("\nCheck 14: NaN diagnosis per metric")
for metric, (ob_c, rec_c, canc_c) in metric_denom.items():
    nan_m = df[df[metric].isna()].copy()
    print(f"\n  {metric}: {len(nan_m)} NaN / {len(df)}")
    if len(nan_m) == 0:
        continue
    zero_funds = (nan_m[ob_c] == 0 if ob_c else True) & (nan_m[rec_c] == 0)
    print(f"    Zero inputs (no funds):      {zero_funds.sum()}")
    if canc_c:
        denom  = (nan_m[ob_c] if ob_c else 0) + nan_m[rec_c] - nan_m[canc_c]
        wiped  = (~zero_funds) & (denom <= 0)
        print(f"    Cancellations wiped funds:   {wiped.sum()}")
    check_cols = [c for c in [ob_c, rec_c, canc_c] if c]
    print(f"    NaN in underlying cols:      {nan_m[check_cols].isna().any(axis=1).sum()}")
    top = nan_m.groupby("district_name").size().sort_values(ascending=False).head(5)
    print(f"    Top 5 districts: {dict(top)}")

# =============================================================================
# CHECK 15: Metric > 1 — breakdown by district
# =============================================================================
print("\nCheck 15: Metric > 1 per metric")
for metric in METRICS:
    over1 = df[df[metric] > 1]
    print(f"\n  {metric}: {len(over1)} rows > 1")
    if len(over1) > 0:
        o1d = (over1.groupby("district_name").size()
               .reset_index(name="over1_count")
               .merge(dist_total, on="district_name"))
        o1d["pct"] = (o1d["over1_count"] / o1d["total_panchayats"] * 100).round(1)
        print(o1d.sort_values("over1_count", ascending=False).head(10).to_string(index=False))
        print(f"  Top 5 highest {metric} values:")
        print(df.nlargest(5, metric)[["district_name","block_name","panchayat_name", metric,
                                      "opening_balance","total_receipts","total_payments"]].to_string(index=False))

# ── Diagnose UI > 1: inflated payments vs understated funds ───────────────────
over1  = df[df["utilization_index"] > 1].copy()
normal = df[(df["utilization_index"] > 0) & (df["utilization_index"] <= 1)].copy()
print(f"\nCheck 15b: UI > 1 diagnosis (n={len(over1)})")
print(f"  {'Column':<30s} {'Med (UI>1)':>12s} {'Med (0<UI<=1)':>14s} {'Ratio':>7s}")
for col in ["opening_balance","total_receipts","total_payments",
            "total_payment_cancellation","total_receipt_cancellation"]:
    mo = over1[col].median(); mn = normal[col].median()
    r  = mo/mn if mn != 0 else np.nan
    print(f"  {col:<30s} {mo:>12,.0f} {mn:>14,.0f} {r:>7.2f}x")

over1["net_payments"]   = over1["total_payments"] - over1["total_payment_cancellation"]
over1["available_funds"]= over1["opening_balance"] + over1["total_receipts"] - over1["total_receipt_cancellation"]
over1["excess"]         = over1["net_payments"] - over1["available_funds"]
print(f"  Median excess spending: {over1['excess'].median():,.0f}")
ob_zero_pct = (over1["opening_balance"] == 0).mean() * 100
print(f"  % of UI>1 panchayats with OB=0: {ob_zero_pct:.1f}%")

bins = [(1,1.5),(1.5,2),(2,5),(5,10),(10,50),(50,float("inf"))]
print(f"\n  {'UI Range':<12s} {'Count':>6s} {'% of >1':>8s} {'% of all':>9s}")
for lo, hi in bins:
    mask = (df["utilization_index"] > lo) & (df["utilization_index"] <= hi)
    cnt  = mask.sum()
    print(f"  {lo}-{hi!s:<8s} {cnt:>6d} {cnt/len(over1)*100:>7.1f}% {cnt/len(df)*100:>8.1f}%")

# =============================================================================
# CHECK 16: Extreme opening balances (1st / 99th percentile concentration)
# =============================================================================
ob   = df["opening_balance"].dropna()
p1   = ob.quantile(0.01); p99 = ob.quantile(0.99)
print(f"\nCheck 16: Opening balance 1st pct={p1:,.2f}, 99th pct={p99:,.2f} (range {ob.min():,.2f}-{ob.max():,.2f})")
for label, subset in [("Bottom 1%", df[df["opening_balance"] <= p1]),
                       ("Top 1%",    df[df["opening_balance"] >= p99])]:
    top_share = subset["district_name"].value_counts(normalize=True).iloc[0] * 100
    top_name  = subset["district_name"].value_counts().index[0]
    print(f"  {label} ({len(subset)} rows) — top district: {top_name} ({top_share:.0f}%)")
    if top_share > 30:
        print(f"  WARNING: {top_share:.0f}% of {label} concentrated in {top_name}")

# =============================================================================
# CHECK 17: Blocks with unusually few or many panchayats
# =============================================================================
block_sizes = df.groupby(["district_name","block_name"]).size().reset_index(name="n_panchayats")
mu_b = block_sizes["n_panchayats"].mean(); sd_b = block_sizes["n_panchayats"].std()
print(f"\nCheck 17: Panchayats per block — min={block_sizes['n_panchayats'].min()}, "
      f"max={block_sizes['n_panchayats'].max()}, median={block_sizes['n_panchayats'].median():.0f}")
tiny  = block_sizes[block_sizes["n_panchayats"] <= 2]
large = block_sizes[block_sizes["n_panchayats"] > mu_b + 3 * sd_b]
print(f"  Blocks with <=2 panchayats: {len(tiny)}" + (" [PASS]" if len(tiny)==0 else " [WARNING]"))
if len(tiny):  print(tiny.to_string(index=False))
print(f"  Blocks >3 SD above mean: {len(large)}" + (" [PASS]" if len(large)==0 else " [WARNING]"))
if len(large): print(large.sort_values("n_panchayats", ascending=False).to_string(index=False))

# =============================================================================
# CHECK 18: Identical financial fingerprints (copy-paste detection)
# =============================================================================
fin_fields   = ["receipts","payments","receipt_cancellation","payment_cancellation","ob_rejected"]
monthly_cols = [f"{m}_{f}" for m in MONTHS for f in fin_fields]
all_zero     = (df[monthly_cols] == 0).all(axis=1)
df_nz        = df[~all_zero]
dupes        = df_nz.duplicated(subset=monthly_cols, keep=False)
n_groups     = df_nz[dupes].groupby(monthly_cols).ngroups if dupes.sum() > 0 else 0
print(f"\nCheck 18: Identical financial fingerprints — {dupes.sum()} rows / {n_groups} groups")
if dupes.sum() == 0:
    print("  PASS")
else:
    sample = df_nz[dupes][["district_name","block_name","panchayat_name","panchayat_code"]].head(10)
    print(sample.to_string(index=False))

# =============================================================================
# CHECK 19: Months with zero activity across an entire district
# =============================================================================
dead = [
    (d, m, len(df[df["district_name"]==d]))
    for d in df["district_name"].unique()
    for m in MONTHS
    if (df.loc[df["district_name"]==d, f"{m}_receipts"] == 0).all()
    and (df.loc[df["district_name"]==d, f"{m}_payments"] == 0).all()
]
if not dead:
    print("\nCheck 19 [PASS]: No district has a completely zero-activity month")
else:
    dead_df = pd.DataFrame(dead, columns=["district","month","n_panchayats"])
    print(f"\nCheck 19 [WARNING]: {len(dead_df)} district-month combinations with zero activity")
    pivot = dead_df.pivot_table(index="district", columns="month",
                                values="n_panchayats", aggfunc="size", fill_value=0)
    pivot = pivot.reindex(columns=[m for m in MONTHS if m in pivot.columns])
    print(pivot.to_string())

# =============================================================================
# CHECK 20: District-level metric distribution (median, mean, CV)
# =============================================================================
print("\nCheck 20: District-level metric distribution")
for metric in METRICS:
    dist_m = df.groupby("district_name")[metric].agg(
        median="median", mean="mean", std="std", count="count",
        n_nan=lambda x: x.isna().sum()
    )
    dist_m["cv"] = (dist_m["std"] / dist_m["mean"].abs()).round(3)
    low   = dist_m[dist_m["median"] < 0.1]
    high  = dist_m[dist_m["median"] >= 1]
    hi_cv = dist_m[dist_m["cv"] > 2]
    print(f"\n  {metric}: low-median districts={len(low)}, median>=1 districts={len(high)}, high-CV districts={len(hi_cv)}")
    if len(low):  print(low[["median","mean","n_nan"]].to_string())
    if len(high): print(high[["median","mean","n_nan"]].sort_values("median",ascending=False).to_string())

# =============================================================================
# CHECK 20b: Negative metric values — root cause
# =============================================================================
print("\nCheck 20b: Negative metric values")
for metric in METRICS:
    neg = df[df[metric] < 0]
    if len(neg) == 0:
        print(f"  {metric}: PASS (no negatives)")
        continue
    tiny_n = (neg[metric] >= -0.01).sum()
    big_n  = (neg[metric] <  -0.01).sum()
    print(f"  {metric}: {len(neg)} negatives (near-zero={tiny_n}, substantial={big_n})")
    if big_n:
        worst = neg.nsmallest(5, metric)[
            ["district_name","panchayat_name","total_payments",
             "total_payment_cancellation","opening_balance","total_receipts", metric]
        ]
        print(worst.to_string(index=False))

# =============================================================================
# CHECK 21: Districts with unusually few panchayats (<2 SD below mean)
# =============================================================================
dist_sizes = df.groupby("district_name").size().reset_index(name="n_panchayats")
mu_d = dist_sizes["n_panchayats"].mean(); sd_d = dist_sizes["n_panchayats"].std()
threshold  = mu_d - 2 * sd_d
small      = dist_sizes[dist_sizes["n_panchayats"] < threshold].sort_values("n_panchayats")
print(f"\nCheck 21: Districts with <2 SD below mean panchayats (threshold={threshold:.0f})")
print("  PASS" if len(small)==0 else f"  WARNING: {len(small)} districts\n{small.to_string(index=False)}")

# =============================================================================
# CHECK 22: All-zero panchayats (completely inactive all year)
# =============================================================================
receipt_cols = [f"{m}_receipts" for m in MONTHS]
payment_cols = [f"{m}_payments" for m in MONTHS]
all_zero_mask = (df[receipt_cols + payment_cols] == 0).all(axis=1)
zero_rows     = df[all_zero_mask]
print(f"\nCheck 22: All-zero panchayats: {len(zero_rows)}/{len(df)}")
if len(zero_rows):
    has_ob = (zero_rows["opening_balance"] > 0).sum()
    print(f"  Of these, {has_ob} have non-zero opening balance (dormant with funds)")
    zbd2 = zero_rows.groupby("district_name").size().reset_index(name="cnt").merge(dist_total, on="district_name")
    zbd2["pct"] = (zbd2["cnt"]/zbd2["total_panchayats"]*100).round(1)
    print(zbd2.sort_values("pct", ascending=False).head(10).to_string(index=False))

# =============================================================================
# CHECK 23: Closing balance reconstruction — flag negatives
# =============================================================================
df["closing_balance"] = (
    df["opening_balance"] + df["total_receipts"] - df["total_receipt_cancellation"]
    - df["total_payments"] + df["total_payment_cancellation"] - df["total_ob_rejected"]
)
neg_cb = df[df["closing_balance"] < 0]
print(f"\nCheck 23: Negative closing balances: {len(neg_cb)}/{len(df)}")
if len(neg_cb):
    print(f"  Range: {neg_cb['closing_balance'].min():,.2f} to {neg_cb['closing_balance'].max():,.2f}")
    nbd = neg_cb.groupby("district_name").size().reset_index(name="cnt").sort_values("cnt",ascending=False)
    print(nbd.head(10).to_string(index=False))
    print(neg_cb.nsmallest(5,"closing_balance")[
        ["district_name","panchayat_name","opening_balance","total_receipts","total_payments","closing_balance"]
    ].to_string(index=False))
else:
    print("  PASS")

# =============================================================================
# CHECK 24: High cancellation ratios (>20%)
# =============================================================================
has_rec = df["total_receipts"] > 0
has_pay = df["total_payments"] > 0
df.loc[has_rec, "receipt_canc_ratio"]  = df.loc[has_rec, "total_receipt_cancellation"]  / df.loc[has_rec, "total_receipts"]
df.loc[has_pay, "payment_canc_ratio"]  = df.loc[has_pay, "total_payment_cancellation"]  / df.loc[has_pay, "total_payments"]
print("\nCheck 24: Cancellation ratios")
for label, col in [("Receipt","receipt_canc_ratio"),("Payment","payment_canc_ratio")]:
    valid = df[col].dropna()
    high  = df[df[col] > 0.2]
    print(f"  {label}: mean={valid.mean():.4f}, median={valid.median():.4f}, "
          f"max={valid.max():.4f}, >20%: {len(high)}")
    if len(high):
        print(high.groupby("district_name").size().sort_values(ascending=False).head(5).to_string())

# =============================================================================
# CHECK 25: Dormant accounts (OB > 0 but zero receipts & payments all year)
# =============================================================================
dormant = df[(df["opening_balance"] > 0) & (df["total_receipts"] == 0) & (df["total_payments"] == 0)]
print(f"\nCheck 25: Dormant panchayats: {len(dormant)}/{len(df)}")
if len(dormant):
    print(f"  Total idle funds: {dormant['opening_balance'].sum():,.2f}")
    dbd = dormant.groupby("district_name").agg(
        n_dormant=("panchayat_name","size"), idle_funds=("opening_balance","sum")
    ).sort_values("n_dormant", ascending=False)
    print(dbd.head(10).to_string())

# =============================================================================
# CHECK 26: March rush — share of annual payments occurring in March
# =============================================================================
monthly_pay   = {m: df[f"{m}_payments"].sum() for m in MONTHS}
total_pay     = sum(monthly_pay.values())
march_share   = monthly_pay["march"] / total_pay * 100
print(f"\nCheck 26: March payment share = {march_share:.1f}%")
if march_share > 25:
    print("  WARNING: March rush very pronounced (>25%)")
print("  Monthly shares:")
for m in MONTHS:
    bar = "#" * int(monthly_pay[m] / total_pay * 100)
    print(f"    {m:>10s}: {monthly_pay[m]/total_pay*100:5.1f}%  {bar}")

df["march_payment_share"] = np.where(
    df["total_payments"] > 0, df["march_payments"] / df["total_payments"], np.nan
)
top_march = df.groupby("district_name")["march_payment_share"].median().sort_values(ascending=False)
print("  Districts with highest median March share (top 10):")
print(top_march.head(10).apply(lambda x: f"{x:.1%}").to_string())

# =============================================================================
# CHECK 27: Prevalence of zeros in total financial columns
# =============================================================================
TOTAL_COLS = ["opening_balance","total_receipts","total_payments",
              "total_receipt_cancellation","total_payment_cancellation","total_ob_rejected"]
n = len(df)
print(f"\nCheck 27: Zero prevalence in total financial columns (n={n})")
print(f"  {'Column':<34s} {'Zeros':>7s} {'%Zero':>7s} {'NaN':>5s}")
for col in TOTAL_COLS:
    nz  = (df[col] == 0).sum()
    nan = df[col].isna().sum()
    print(f"  {col:<34s} {nz:>7d} {nz/n*100:>6.1f}% {nan:>5d}")

# Combo: all three key columns zero
all3 = df[(df["opening_balance"]==0) & (df["total_receipts"]==0) & (df["total_payments"]==0)]
print(f"\n  Panchayats with OB=0 AND receipts=0 AND payments=0: {len(all3)}/{n} ({len(all3)/n*100:.1f}%)")

# =============================================================================
# EXPORT cleaned dataframe
# =============================================================================
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nExported cleaned data to: {OUTPUT_FILE}")
print(f"  Rows: {len(df)} | Columns: {len(df.columns)}")
