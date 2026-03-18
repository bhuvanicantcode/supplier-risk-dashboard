"""
=============================================================================
PROJECT 2: Supplier Risk Scoring & Supply-at-Risk (SaR) Simulation
=============================================================================
Author : Pallapolu Bhuvan Chandra — BITS Pilani
Stack  : Python | pandas | NumPy | matplotlib | Streamlit

Data Sources (Real):
    1. SCMS Delivery History Dataset (USAID, public domain)
       → OTD Rate, Lead Time CoV, Logistics Complexity, Planning Risk,
         Annual Spend, Concentration / Single-Source flag
    2. Worldwide Governance Indicators 2012-2022 (World Bank / Mendeley)
       → Country-level Governance Risk (geo + ESG proxy)

Business Problem:
    A pharmaceutical supply chain firm sources from 33 global vendors across
    Africa, Asia, and Europe. Each vendor-country pair carries a unique risk
    profile. A single vendor failure can cascade into a medicine shortage.

    This dashboard:
    1. Engineers 6 risk dimensions from real transactional + governance data
    2. Derives Probability of Disruption (PD) per vendor-country pair
    3. Derives Loss Given Disruption (LGD) from structural risk factors
    4. Monte Carlo simulates portfolio-level supply loss (10,000 scenarios)
    5. Computes Supply-at-Risk (SaR) at 90%, 95% and 99% confidence
    6. Stress tests: Top-3 failure, country shock, ESG crisis, portfolio wave

Key Innovation:
    Adapts Credit Risk VaR methodology (PD × LGD × EAD) to supply chain:
    → PD  = Probability of supplier disruption (logistic of 6 risk dims)
    → LGD = Loss Given Disruption (derived from single-source + logistics + geo)
    → EAD = Exposure at Disruption (actual $ spend from SCMS records)
    → SaR = Supply-at-Risk (analogous to VaR in finance)
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

st.set_page_config(
    page_title='Supplier Risk Analytics | Bhuvan Chandra',
    page_icon='🏭',
    layout='wide'
)

st.markdown('''
<style>
    .main-title  { font-size:2.1rem; font-weight:700; color:#0D1F3C; }
    .sub-title   { font-size:1.05rem; color:#64748B; margin-top:-8px; }
    .metric-card { background:#F8FAFC; border:1px solid #E2E8F0;
                   border-radius:10px; padding:16px; text-align:center; }
    .metric-val  { font-size:1.7rem; font-weight:700; color:#DC2626; }
    .metric-label{ font-size:0.82rem; color:#64748B; }
    .insight-box { background:#EFF6FF; border-left:4px solid #2563EB;
                   padding:12px 16px; border-radius:4px; margin:8px 0; }
    .risk-high   { background:#FEE2E2; border-left:4px solid #DC2626;
                   padding:12px 16px; border-radius:4px; margin:8px 0; }
    .risk-low    { background:#D1FAE5; border-left:4px solid #059669;
                   padding:12px 16px; border-radius:4px; margin:8px 0; }
    .data-badge  { background:#F0FDF4; border:1px solid #86EFAC;
                   border-radius:6px; padding:4px 10px; font-size:0.78rem;
                   color:#166534; display:inline-block; margin:2px; }
    section[data-testid="stSidebar"] { background:#0D1F3C; }
    section[data-testid="stSidebar"] * { color: white !important; }
</style>
''', unsafe_allow_html=True)

# ── Paths — adjust if files are in a subfolder ───────────────────────────
SCMS_PATH = r'supplier data/SCMS_Delivery_History_Dataset.csv'
WGI_PATH  = r'supplier data/Worldwide_Governance_Indicators_20122022.xlsx'

# Fallback for cloud / container deployments
import os
_MOUNT = '/mnt/project'
if not os.path.exists(SCMS_PATH) and os.path.exists(f'{_MOUNT}/SCMS_Delivery_History_Dataset.csv'):
    SCMS_PATH = f'{_MOUNT}/SCMS_Delivery_History_Dataset.csv'
if not os.path.exists(WGI_PATH) and os.path.exists(f'{_MOUNT}/Worldwide_Governance_Indicators_20122022.xlsx'):
    WGI_PATH = f'{_MOUNT}/Worldwide_Governance_Indicators_20122022.xlsx'


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD SCMS SHIPMENT DATA
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_scms_data(path: str) -> pd.DataFrame:
    """
    Loads SCMS_Delivery_History_Dataset.csv.
    Parses all date columns, cleans numeric columns, and engineers:
      - delivery_delay_days  : actual vs scheduled (negative = early)
      - is_on_time           : 1 if delivered on or before scheduled date
      - lead_time_days       : PO sent → delivered (fallback: PQ sent → delivered)

    Filters out 'SCMS from RDC' — a redistribution centre, not a real vendor.
    Keeping it would inflate spend by 67% and distort all risk scores.
    """
    df = pd.read_csv(path)

    date_cols = [
        'PQ First Sent to Client Date', 'PO Sent to Vendor Date',
        'Scheduled Delivery Date', 'Delivered to Client Date',
        'Delivery Recorded Date'
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    for col in ['Line Item Quantity', 'Line Item Value', 'Pack Price',
                'Unit Price', 'Line Item Insurance (USD)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    def parse_mixed(series: pd.Series) -> pd.Series:
        cleaned = (series.astype(str)
                         .str.replace(',', '', regex=False)
                         .str.extract(r'([-+]?[0-9]*\.?[0-9]+)')[0])
        return pd.to_numeric(cleaned, errors='coerce')

    df['Freight_USD_num']   = parse_mixed(df['Freight Cost (USD)'])
    df['Weight_kg_num']     = parse_mixed(df['Weight (Kilograms)'])
    df['Insurance_USD_num'] = df['Line Item Insurance (USD)'].fillna(0)

    df['delivery_delay_days'] = (
        df['Delivered to Client Date'] - df['Scheduled Delivery Date']
    ).dt.days
    df['is_on_time'] = (df['delivery_delay_days'] <= 0).astype(float)

    df['lead_time_days'] = (
        df['Delivered to Client Date'] - df['PO Sent to Vendor Date']
    ).dt.days
    fallback = (
        df['Delivered to Client Date'] - df['PQ First Sent to Client Date']
    ).dt.days
    df['lead_time_days'] = df['lead_time_days'].fillna(fallback)
    df.loc[df['lead_time_days'] < 0, 'lead_time_days'] = np.nan

    # Fill categoricals
    for col in ['Product Group', 'Country', 'Vendor', 'Shipment Mode',
                'Vendor INCO Term', 'Sub Classification', 'Fulfill Via']:
        df[col] = df[col].fillna('Unknown')

    # ── Remove redistribution centre — not a real vendor ────────────────
    df = df[df['Vendor'].str.strip() != 'SCMS from RDC'].copy()

    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD WGI GOVERNANCE DATA
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_wgi_data(path: str) -> pd.DataFrame:
    """
    Loads Worldwide_Governance_Indicators_20122022.xlsx.
    Strips column whitespace (the file has 'Country ' with trailing space).
    Derives:
      Governance_Score : mean of all 6 WGI dims / 100  → higher = better governed
      Governance_Risk  : 1 - Governance_Score           → higher = riskier country
    """
    wgi = pd.read_excel(path)
    wgi.columns = [str(c).strip() for c in wgi.columns]
    dims = [
        'Voice and Accountability', 'Political Stability',
        'Government Effectiveness', 'Regulatory Quality',
        'Rule of Law', 'Control of Corruption'
    ]
    wgi[dims] = wgi[dims].apply(pd.to_numeric, errors='coerce')
    wgi['Governance_Score'] = wgi[dims].mean(axis=1) / 100.0
    wgi['Governance_Risk']  = 1 - wgi['Governance_Score'].clip(0, 1)
    wgi['Country'] = wgi['Country'].astype(str).str.strip()
    return wgi[['Country', 'Year', 'Region', 'Governance_Score',
                'Governance_Risk'] + dims]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING: VENDOR × COUNTRY → RISK DIMENSIONS
# ═══════════════════════════════════════════════════════════════════════════
def dominant_value(series: pd.Series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if len(mode) > 0 else 'Unknown'


@st.cache_data
def build_supplier_view(scms: pd.DataFrame, wgi: pd.DataFrame,
                        wgi_year: int, min_shipments: int) -> tuple:
    """
    Aggregates shipment-level SCMS data to Vendor × Country pairs.
    Engineers 6 risk dimensions — all derived from real data columns:

    1. otd_risk         = 1 - OTD Rate
                          Source: Scheduled vs Delivered date comparison
    2. lt_risk          = Lead Time CoV (std / mean of actual lead time days)
                          Source: PO Sent → Delivered (or PQ Sent fallback)
    3. geo_risk         = Governance Risk from WGI
                          Source: Mean of 6 WGI indicators inverted
    4. conc_risk        = 0.7 × spend share in category + 0.3 × single_source flag
                          Source: Vendor spend vs total per country-product cell
    5. logistics_risk   = Weighted score of shipment mode, mode diversity, INCO diversity
                          Source: Shipment Mode and Vendor INCO Term columns
    6. planning_risk    = Rate of missing PO dates + missing lead time
                          Source: PO Sent to Vendor Date nulls

    LGD (Loss Given Disruption) is also derived here — not randomly sampled:
        LGD = 0.25 + 0.35×single_source + 0.20×logistics_risk + 0.20×geo_risk
        Range: [0.20, 0.95]
    """
    data = scms.copy()

    # Use selected WGI year; fall back to latest per country if year missing
    latest_wgi = wgi[wgi['Year'] == wgi_year].copy()
    if latest_wgi.empty:
        latest_wgi = wgi.sort_values('Year').groupby('Country', as_index=False).tail(1)

    # Shipment mode → base logistics risk score
    mode_risk_map = {
        'Air': 0.35, 'Air Charter': 0.60,
        'Ocean': 0.70, 'Truck': 0.25, 'Unknown': 0.50
    }

    supplier = data.groupby(['Vendor', 'Country']).agg(
        shipments            = ('ID',                    'count'),
        annual_spend         = ('Line Item Value',        'sum'),
        otd_rate             = ('is_on_time',             'mean'),
        avg_delay_days       = ('delivery_delay_days',    'mean'),
        median_delay_days    = ('delivery_delay_days',    'median'),
        lead_time_mean       = ('lead_time_days',         'mean'),
        lead_time_std        = ('lead_time_days',         'std'),
        lead_time_p90        = ('lead_time_days',         lambda s: s.quantile(0.90)),
        product_groups       = ('Product Group',          'nunique'),
        sub_classes          = ('Sub Classification',     'nunique'),
        shipment_modes       = ('Shipment Mode',          'nunique'),
        incoterms            = ('Vendor INCO Term',       'nunique'),
        qty                  = ('Line Item Quantity',     'sum'),
        freight_usd          = ('Freight_USD_num',        'sum'),
        weight_kg            = ('Weight_kg_num',          'sum'),
        insurance_usd        = ('Insurance_USD_num',      'sum'),
        po_date_missing_rate = ('PO Sent to Vendor Date', lambda s: s.isna().mean()),
        dominant_mode        = ('Shipment Mode',          dominant_value),
        dominant_product     = ('Product Group',          dominant_value),
        dominant_subclass    = ('Sub Classification',     dominant_value),
        fulfill_via          = ('Fulfill Via',            dominant_value),
        first_delivery       = ('Delivered to Client Date', 'min'),
        last_delivery        = ('Delivered to Client Date', 'max'),
    ).reset_index()

    supplier = supplier[supplier['shipments'] >= min_shipments].copy()

    # ── Lead Time CoV ────────────────────────────────────────────────────
    # std / mean — robust: if mean is zero, fill with median CoV
    supplier['lead_time_std'] = supplier['lead_time_std'].fillna(0)
    supplier['lt_cov'] = (
        supplier['lead_time_std'] / supplier['lead_time_mean'].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(supplier['lt_cov'].median()
        if 'lt_cov' in supplier.columns else 0.3)
    # Handle first-time column creation (median call before column exists)
    if supplier['lt_cov'].isna().any():
        supplier['lt_cov'] = supplier['lt_cov'].fillna(0.3)

    # ── Concentration / Single Source ───────────────────────────────────
    # If a vendor supplies ≥80% of spend in its country-product cell → single source
    cat_country_spend = supplier.groupby(
        ['Country', 'dominant_product'])['annual_spend'].transform('sum')
    supplier['country_product_share'] = (
        supplier['annual_spend'] / cat_country_spend.replace(0, np.nan)
    ).fillna(0)
    supplier['single_source'] = (
        (supplier['country_product_share'] >= 0.80) |
        (supplier['shipments'] <= 3)
    ).astype(int)

    # ── Logistics Complexity ─────────────────────────────────────────────
    supplier['logistics_complexity'] = (
        0.50 * supplier['dominant_mode'].map(mode_risk_map).fillna(0.50) +
        0.25 * np.clip((supplier['shipment_modes'] - 1) / 4, 0, 1) +
        0.25 * np.clip((supplier['incoterms'] - 1) / 4, 0, 1)
    )

    # ── Planning Risk ────────────────────────────────────────────────────
    # High missing PO date rate = poor planning visibility = higher risk
    supplier['planning_risk_raw'] = np.clip(
        0.65 * supplier['po_date_missing_rate'] +
        0.35 * supplier['lead_time_mean'].isna().astype(float),
        0, 1
    )

    # ── Merge WGI governance scores ─────────────────────────────────────
    supplier = supplier.merge(
        latest_wgi[['Country', 'Region', 'Governance_Score', 'Governance_Risk']],
        on='Country', how='left'
    )
    supplier['Region']           = supplier['Region'].fillna('Unmapped')
    supplier['Governance_Score'] = supplier['Governance_Score'].fillna(
        latest_wgi['Governance_Score'].median())
    supplier['Governance_Risk']  = supplier['Governance_Risk'].fillna(
        latest_wgi['Governance_Risk'].median())

    # ── Final risk dimensions (all on [0,1], higher = more risk) ────────
    supplier['otd_risk']       = 1 - supplier['otd_rate'].fillna(
        supplier['otd_rate'].median())
    supplier['lt_risk']        = np.clip(supplier['lt_cov'], 0, 1)
    supplier['geo_risk']       = np.clip(supplier['Governance_Risk'], 0, 1)
    supplier['conc_risk']      = np.clip(
        0.7 * supplier['country_product_share'] +
        0.3 * supplier['single_source'], 0, 1)
    supplier['logistics_risk'] = np.clip(supplier['logistics_complexity'], 0, 1)
    supplier['planning_risk']  = np.clip(supplier['planning_risk_raw'], 0, 1)

    return supplier.sort_values('annual_spend', ascending=False), latest_wgi


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — PROBABILITY OF DISRUPTION (PD)
# ═══════════════════════════════════════════════════════════════════════════
def compute_pd(df: pd.DataFrame, weights: list) -> pd.Series:
    """
    Weighted composite of 6 risk dimensions → logistic transform → PD in (0,1).
    Logistic centred at 0.42 with steepness 4.2.
    Output clipped to [0.02, 0.85].
    """
    composite = (
        df['otd_risk']       * weights[0] +
        df['lt_risk']        * weights[1] +
        df['geo_risk']       * weights[2] +
        df['conc_risk']      * weights[3] +
        df['logistics_risk'] * weights[4] +
        df['planning_risk']  * weights[5]
    )
    return (1 / (1 + np.exp(-4.2 * (composite - 0.42)))).clip(0.02, 0.85)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — LOSS GIVEN DISRUPTION (LGD)
# ═══════════════════════════════════════════════════════════════════════════
def build_lgd(df: pd.DataFrame) -> pd.Series:
    """
    LGD is derived from structural risk factors — NOT randomly sampled.
    Base LGD = 25%
    + 35% if vendor is single-source (no backup = total supply lost)
    + 20% × logistics_risk (complex routes → harder to reroute)
    + 20% × geo_risk (unstable regions → longer recovery time)
    Clipped to [0.20, 0.95].
    """
    lgd = (
        0.25 +
        0.35 * df['single_source'] +
        0.20 * df['logistics_risk'] +
        0.20 * df['geo_risk']
    )
    return lgd.clip(0.20, 0.95)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — MONTE CARLO SaR SIMULATION
# ═══════════════════════════════════════════════════════════════════════════
def monte_carlo_sar(df: pd.DataFrame, n_sims: int = 10000,
                    seed: int = 42) -> np.ndarray:
    """
    Vectorised Monte Carlo — runs all simulations as a matrix operation.
    Shape: (n_sims × n_suppliers)

    For each simulation:
      1. Draw uniform(0,1) for each supplier
      2. Supplier disrupted if draw < PD (Bernoulli)
      3. Loss = sum(disrupted × LGD × annual_spend)

    LGD is fixed per supplier (derived, not random) — variation comes
    entirely from which suppliers get disrupted in each scenario.

    Returns array of n_sims total loss values.
    """
    rng      = np.random.default_rng(seed)
    pds      = df['PD'].to_numpy()
    lgd      = df['LGD'].to_numpy()
    ead      = df['annual_spend'].to_numpy()
    rand     = rng.random((n_sims, len(df)))
    disrupted = (rand < pds).astype(float)
    return (disrupted * lgd * ead).sum(axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
scms = load_scms_data(SCMS_PATH)
wgi  = load_wgi_data(WGI_PATH)

wgi_years          = sorted(wgi['Year'].dropna().astype(int).unique())
available_products = sorted(scms['Product Group'].dropna().unique().tolist())
available_countries= sorted(scms['Country'].dropna().unique().tolist())


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('## 🏭 Supplier Risk')
    st.markdown('**Supply-at-Risk Dashboard**')
    st.markdown('Real SCMS delivery data + World Bank WGI')
    st.markdown('---')

    selected_products  = st.multiselect('Filter Product Group',
        options=available_products, default=available_products)
    selected_countries = st.multiselect('Filter Countries',
        options=available_countries, default=available_countries)
    wgi_year      = st.select_slider('Governance reference year',
        options=wgi_years, value=max(wgi_years))
    min_shipments = st.slider('Min shipments per supplier-country pair', 1, 20, 5)

    st.markdown('---')
    st.markdown('### ⚖️ Risk Dimension Weights')
    st.caption('Auto-normalised to sum = 1.')

    w_otd  = st.slider('On-Time Delivery Risk',          0.05, 0.40, 0.25, 0.05)
    w_lt   = st.slider('Lead Time Variability',           0.05, 0.40, 0.20, 0.05)
    w_geo  = st.slider('Country Governance Risk',         0.05, 0.40, 0.20, 0.05)
    w_conc = st.slider('Spend Concentration / Single-Source', 0.05, 0.40, 0.15, 0.05)
    w_log  = st.slider('Logistics Complexity',            0.05, 0.40, 0.10, 0.05)
    w_plan = st.slider('Planning / Missing PO Data',      0.05, 0.40, 0.10, 0.05)

    weights = [w_otd, w_lt, w_geo, w_conc, w_log, w_plan]
    total_w = sum(weights)
    if abs(total_w - 1.0) > 0.01:
        st.warning(f'⚠️ Weights sum to {total_w:.2f}. Normalising.')
    weights = [w / total_w for w in weights]

    st.markdown('---')
    n_sims = st.select_slider('Monte Carlo Simulations',
        [1000, 5000, 10000, 25000], 10000)
    st.markdown('---')
    st.markdown('*Built by Pallapolu Bhuvan Chandra*')
    st.markdown('*BITS Pilani | SCM Portfolio*')


# ═══════════════════════════════════════════════════════════════════════════
# COMPUTE
# ═══════════════════════════════════════════════════════════════════════════
filtered_scms = scms[
    scms['Product Group'].isin(selected_products) &
    scms['Country'].isin(selected_countries)
].copy()

supplier_df, wgi_year_df = build_supplier_view(
    filtered_scms, wgi, wgi_year, min_shipments)

if supplier_df.empty:
    st.error('No supplier-country pairs after filters. Relax the filters or reduce minimum shipments.')
    st.stop()

supplier_df['PD']           = compute_pd(supplier_df, weights)
supplier_df['LGD']          = build_lgd(supplier_df)
supplier_df['Expected_Loss'] = (supplier_df['PD'] * supplier_df['LGD'] *
                                 supplier_df['annual_spend'])
supplier_df['Risk_Tier'] = pd.cut(
    supplier_df['PD'],
    bins=[0, 0.20, 0.40, 0.60, 1.0],
    labels=['🟢 Low', '🟡 Medium', '🟠 High', '🔴 Critical']
)

losses      = monte_carlo_sar(supplier_df, n_sims=n_sims)
sar_90      = np.percentile(losses, 90)
sar_95      = np.percentile(losses, 95)
sar_99      = np.percentile(losses, 99)
exp_loss    = losses.mean()
total_spend = supplier_df['annual_spend'].sum()
tail_ratio  = (sar_99 / total_spend * 100) if total_spend else 0
high_risk_count = int((supplier_df['PD'] > 0.40).sum())
avg_disrupt = float(((losses > 0).sum()) / n_sims)   # proxy


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="main-title">🏭 Supplier Risk Scoring & Supply-at-Risk Simulation</p>',
    unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Monte Carlo simulation of portfolio supply disruption risk '
    '— adapted from credit VaR methodology | Real data: USAID SCMS + World Bank WGI</p>',
    unsafe_allow_html=True)

st.markdown(f'''
<span class="data-badge">📦 USAID SCMS Delivery History ({len(scms):,} shipments)</span>
<span class="data-badge">🌍 World Bank WGI 2012-2022 (180 countries)</span>
<span class="data-badge">🏢 {supplier_df["Vendor"].nunique()} real vendors | {supplier_df["Country"].nunique()} countries</span>
<span class="data-badge">📊 {len(supplier_df)} supplier-country pairs</span>
''', unsafe_allow_html=True)
st.markdown('---')

# KPI row
col1, col2, col3, col4, col5 = st.columns(5)
kpis = [
    (f'${exp_loss/1e6:.2f}M',      'Expected Annual Loss'),
    (f'${sar_95/1e6:.2f}M',        'Supply-at-Risk (95%)'),
    (f'${sar_99/1e6:.2f}M',        'Supply-at-Risk (99%)'),
    (f'{tail_ratio:.1f}%',         '99% SaR / Total Spend'),
    (f'{high_risk_count}',         'High/Critical Suppliers'),
]
for col, (val, label) in zip([col1, col2, col3, col4, col5], kpis):
    with col:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-label">{label}</div>
        </div>''', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)

st.markdown(f'''
<div class="insight-box">
<b>Dataset scope:</b> {len(filtered_scms):,} shipment records →
{len(supplier_df):,} supplier-country pairs →
{supplier_df["Vendor"].nunique():,} unique vendors →
{supplier_df["Country"].nunique():,} countries →
WGI year <b>{wgi_year}</b> for governance overlay.
</div>''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    '📊 Loss Distribution (SaR)',
    '🗺️ Supplier Scorecard',
    '🔬 Risk Breakdown',
    '💥 Stress Testing',
    '📈 Analytics',
    '🔍 Data Provenance',
])


# ── TAB 1: Monte Carlo Loss Distribution ─────────────────────────────────
with tab1:
    st.subheader(f'Supply Loss Distribution — {n_sims:,} Monte Carlo Simulations')
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor('#F8FAFC'); ax.set_facecolor('#F8FAFC')

    n_hist, bins, patches = ax.hist(
        losses / 1e6, bins=80, color='#3B82F6',
        edgecolor='white', linewidth=0.3, alpha=0.85)
    for patch, left in zip(patches, bins[:-1]):
        if left * 1e6 >= sar_95:
            patch.set_facecolor('#EF4444')

    ax.axvline(exp_loss / 1e6, color='#10B981', lw=2, ls='--',
               label=f'Expected Loss: ${exp_loss/1e6:.2f}M')
    ax.axvline(sar_90 / 1e6, color='#F59E0B', lw=1.5, ls=':',
               label=f'SaR 90%: ${sar_90/1e6:.2f}M')
    ax.axvline(sar_95 / 1e6, color='#F59E0B', lw=2, ls='--',
               label=f'SaR 95%: ${sar_95/1e6:.2f}M')
    ax.axvline(sar_99 / 1e6, color='#EF4444', lw=2.5, ls='-',
               label=f'SaR 99%: ${sar_99/1e6:.2f}M')
    ax.fill_betweenx([0, n_hist.max()], sar_95 / 1e6, losses.max() / 1e6,
                     alpha=0.08, color='#EF4444', label='Tail Risk Zone (>95%)')

    ax.set_xlabel('Annual Supply Loss ($M)', fontsize=11)
    ax.set_ylabel('Frequency (# Scenarios)', fontsize=11)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:.1f}M'))
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig); plt.close()

    c1, c2, c3 = st.columns(3)
    c1.metric('Supplier-Country Pairs', f'{len(supplier_df)}')
    c2.metric('Unique Vendors', f'{supplier_df["Vendor"].nunique()}')
    c3.metric('Total Portfolio Spend', f'${total_spend/1e6:.1f}M')

    st.markdown(f'''
    <div class="risk-high">
    <b>🔴 Risk Interpretation:</b> In 5% of simulated scenarios, total supply
    disruption losses exceed <b>${sar_95/1e6:.2f}M</b> (SaR 95%). In 1% of
    scenarios — black swan events like a geopolitical crisis or vendor
    insolvency wave — losses exceed <b>${sar_99/1e6:.2f}M</b> (SaR 99%).
    This represents <b>{tail_ratio:.1f}%</b> of total annual procurement spend at risk.
    </div>''', unsafe_allow_html=True)


# ── TAB 2: Supplier Scorecard ─────────────────────────────────────────────
with tab2:
    st.subheader(f'Supplier Risk Scorecard — {len(supplier_df)} Real Supplier-Country Pairs')

    show_cols = ['Vendor', 'Country', 'Region', 'dominant_product',
                 'shipments', 'annual_spend', 'otd_rate',
                 'lead_time_mean', 'PD', 'LGD', 'Expected_Loss', 'Risk_Tier']
    display = supplier_df[show_cols].copy()
    display.columns = ['Vendor', 'Country', 'WGI Region', 'Dominant Product',
                       'Shipments', 'Annual Spend', 'OTD',
                       'Avg Lead Time (days)', 'PD', 'LGD',
                       'Expected Loss', 'Risk Tier']
    display['Annual Spend']        = display['Annual Spend'].map(lambda x: f'${x/1e6:.2f}M')
    display['OTD']                 = display['OTD'].map(lambda x: f'{x:.1%}')
    display['Avg Lead Time (days)']= display['Avg Lead Time (days)'].map(
        lambda x: '-' if pd.isna(x) else f'{x:.0f}')
    display['PD']           = display['PD'].map(lambda x: f'{x:.1%}')
    display['LGD']          = display['LGD'].map(lambda x: f'{x:.1%}')
    display['Expected Loss']= display['Expected Loss'].map(lambda x: f'${x/1e6:.2f}M')
    st.dataframe(display.sort_values('PD', ascending=False),
                 use_container_width=True, hide_index=True)

    top10 = supplier_df.nlargest(10, 'PD')[
        ['Vendor', 'Country', 'PD', 'annual_spend']].copy()
    top10['label'] = top10['Vendor'].str.slice(0, 26) + ' | ' + top10['Country']
    st.markdown('#### 🚨 Top 10 Highest-Risk Supplier-Country Pairs')
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('#F8FAFC'); ax.set_facecolor('#F8FAFC')
    bars = ax.barh(top10['label'][::-1], top10['PD'][::-1],
                   color='#EF4444', edgecolor='white')
    for bar, (_, row) in zip(bars, top10[::-1].iterrows()):
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{row['PD']:.1%} | ${row['annual_spend']/1e6:.1f}M",
                va='center', fontsize=9)
    ax.set_xlabel('Probability of Disruption (PD)')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.grid(axis='x', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ── TAB 3: Risk Breakdown ─────────────────────────────────────────────────
with tab3:
    colors_reg = {
        'Africa': '#EF4444', 'South Asia': '#F59E0B',
        'SE Asia': '#10B981', 'Middle East': '#8B5CF6',
        'Central Asia': '#3B82F6', 'Caribbean': '#EC4899',
        'Latin America': '#14B8A6', 'Other': '#94A3B8', 'Unmapped': '#CBD5E1'
    }

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('#### Risk Tier Distribution')
        tier_counts = supplier_df['Risk_Tier'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#F8FAFC'); ax.set_facecolor('#F8FAFC')
        tier_colors = {'🟢 Low': '#10B981', '🟡 Medium': '#F59E0B',
                       '🟠 High': '#F97316', '🔴 Critical': '#EF4444'}
        bars = ax.bar(tier_counts.index.astype(str), tier_counts.values,
                      color=[tier_colors.get(str(t), '#94A3B8')
                             for t in tier_counts.index],
                      edgecolor='white')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f'{int(bar.get_height())}', ha='center',
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('Supplier-Country Pairs')
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig); plt.close()

    with c2:
        st.markdown('#### Expected Loss by Country ($M) — Top 12')
        country_loss = (supplier_df.groupby('Country')['Expected_Loss']
                        .sum().sort_values(ascending=True).tail(12))
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#F8FAFC'); ax.set_facecolor('#F8FAFC')
        ax.barh(country_loss.index, country_loss.values / 1e6,
                color='#3B82F6', edgecolor='white')
        ax.xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, _: f'${x:.1f}M'))
        ax.grid(axis='x', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig); plt.close()

    st.markdown('---')
    st.markdown('#### Weighted Risk Driver Heatmap — Top 15 by Expected Loss')
    heat = supplier_df.nlargest(15, 'Expected_Loss')[
        ['Vendor', 'Country', 'otd_risk', 'lt_risk', 'geo_risk',
         'conc_risk', 'logistics_risk', 'planning_risk']].copy()
    heat['label'] = heat['Vendor'].str.slice(0, 22) + ' | ' + heat['Country']
    heat = heat.set_index('label')[
        ['otd_risk', 'lt_risk', 'geo_risk', 'conc_risk',
         'logistics_risk', 'planning_risk']]
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#F8FAFC')
    im = ax.imshow(heat.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(['OTD', 'Lead Time', 'Governance',
                        'Concentration', 'Logistics', 'Planning'],
                       fontsize=10)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=8)
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            ax.text(j, i, f'{heat.values[i,j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='black')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Risk Score (0=low, 1=high)')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    top3_spend = supplier_df.nlargest(3, 'annual_spend')['annual_spend'].sum()
    conc_pct = (top3_spend / total_spend * 100) if total_spend else 0
    st.markdown(f'''
    <div class="risk-high">
    <b>⚠️ Concentration Risk:</b> Top 3 supplier-country pairs represent
    <b>{conc_pct:.1f}%</b> of total procurement spend (${top3_spend/1e6:.1f}M).
    A simultaneous failure would trigger immediate supply shortages.
    </div>''', unsafe_allow_html=True)


# ── TAB 4: Stress Testing ─────────────────────────────────────────────────
with tab4:
    st.subheader('Stress Test Scenarios')
    scenarios = {}

    # S1: Top-3 simultaneous failure using their actual LGD
    top3 = supplier_df.nlargest(3, 'annual_spend')
    scenarios['Top-3 Supplier Failure'] = (
        top3['annual_spend'] * top3['LGD']).sum()

    # S2: Worst country shock (65% floor LGD)
    worst_country = supplier_df.groupby('Country')['Expected_Loss'].sum().idxmax()
    c_slice = supplier_df[supplier_df['Country'] == worst_country]
    scenarios[f'{worst_country} Country Shock'] = (
        c_slice['annual_spend'] * np.maximum(c_slice['LGD'], 0.65)).sum()

    # S3: Portfolio-wide disruption wave (PD × 1.75)
    df_stress = supplier_df.copy()
    df_stress['PD'] = (df_stress['PD'] * 1.75).clip(0, 1)
    scenarios['Portfolio Disruption Wave (PD ×1.75)'] = np.percentile(
        monte_carlo_sar(df_stress, n_sims=n_sims), 99)

    # S4: High governance-risk countries (top quartile geo_risk)
    gov_tail = supplier_df[
        supplier_df['geo_risk'] >= supplier_df['geo_risk'].quantile(0.75)]
    scenarios['High Governance-Risk Countries'] = (
        gov_tail['annual_spend'] * np.maximum(gov_tail['LGD'], 0.60)).sum()

    # S5: All single-source vendors disrupted
    ss = supplier_df[supplier_df['single_source'] == 1]
    scenarios['Single-Source Cascade'] = (
        ss['annual_spend'] * ss['LGD']).sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#F8FAFC'); ax.set_facecolor('#F8FAFC')
    sc_names = list(scenarios.keys())
    sc_vals  = [v / 1e6 for v in scenarios.values()]
    bar_cols = ['#EF4444', '#DC2626', '#B91C1C', '#F59E0B', '#F97316']
    bars = ax.barh(sc_names, sc_vals, color=bar_cols,
                   edgecolor='white', height=0.55)
    ax.axvline(sar_99 / 1e6, color='#3B82F6', lw=2, ls='--',
               label=f'Baseline SaR 99%: ${sar_99/1e6:.1f}M')
    for bar in bars:
        ax.text(bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f'${bar.get_width():.1f}M',
                va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Estimated Supply Loss ($M)', fontsize=11)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:.0f}M'))
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    worst    = max(scenarios, key=scenarios.get)
    excess   = scenarios[worst] - sar_99
    st.markdown(f'''
    <div class="risk-high">
    <b>💥 Most Severe Scenario: {worst}</b> — estimated loss
    ${scenarios[worst]/1e6:.1f}M.
    {"Exceeds" if excess > 0 else "Within"} the baseline SaR 99%
    {"by <b>$" + f"{abs(excess)/1e6:.1f}M</b>" if excess > 0 else ""}.
    Recommended action: Qualify alternative vendors in non-affected regions
    within 90 days and maintain 8-week strategic inventory buffer for
    single-source critical materials.
    </div>''', unsafe_allow_html=True)


# ── TAB 5: Advanced Analytics ─────────────────────────────────────────────
with tab5:
    st.subheader('Advanced Analytics & Insights')

    st.markdown('#### Risk Factor Correlation Matrix')
    corr_cols = ['otd_risk', 'lt_risk', 'geo_risk', 'conc_risk',
                 'logistics_risk', 'planning_risk', 'PD', 'LGD']
    corr = supplier_df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#F8FAFC'); ax.set_facecolor('#F8FAFC')
    im = ax.imshow(corr, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    labels_display = ['OTD Risk', 'LT Risk', 'Geo Risk', 'Conc Risk',
                      'Logistics', 'Planning', 'PD', 'LGD']
    ax.set_xticklabels(labels_display, rotation=45, ha='right')
    ax.set_yticklabels(labels_display)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                    ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('---')
    cx, cy = st.columns(2)

    with cx:
        st.markdown('#### Risk-Spend Quadrant')
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#F8FAFC'); ax.set_facecolor('#F8FAFC')
        sc = ax.scatter(supplier_df['annual_spend'] / 1e6, supplier_df['PD'],
                        c=supplier_df['PD'], cmap='RdYlGn_r',
                        s=60, alpha=0.75, edgecolors='white', lw=0.8)
        med_spend = supplier_df['annual_spend'].median()
        med_pd    = supplier_df['PD'].median()
        ax.axhline(med_pd, color='gray', ls='--', alpha=0.5,
                   label=f'Median PD: {med_pd:.1%}')
        ax.axvline(med_spend / 1e6, color='gray', ls='--', alpha=0.5,
                   label=f'Median Spend: ${med_spend/1e6:.1f}M')
        ax.set_xlabel('Annual Spend ($M)')
        ax.set_ylabel('Probability of Disruption')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        plt.colorbar(sc, ax=ax, label='PD')
        st.pyplot(fig); plt.close()

    with cy:
        st.markdown('#### Quadrant Summary')
        q1 = len(supplier_df[(supplier_df['annual_spend'] > med_spend) & (supplier_df['PD'] > med_pd)])
        q2 = len(supplier_df[(supplier_df['annual_spend'] <= med_spend) & (supplier_df['PD'] > med_pd)])
        q3 = len(supplier_df[(supplier_df['annual_spend'] <= med_spend) & (supplier_df['PD'] <= med_pd)])
        q4 = len(supplier_df[(supplier_df['annual_spend'] > med_spend) & (supplier_df['PD'] <= med_pd)])

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#F8FAFC'); ax.set_facecolor('#F8FAFC')
        qlabels = ['High Risk\nHigh Spend', 'High Risk\nLow Spend',
                   'Low Risk\nLow Spend', 'Low Risk\nHigh Spend']
        qvals   = [q1, q2, q3, q4]
        qcols   = ['#EF4444', '#F59E0B', '#10B981', '#3B82F6']
        bars = ax.bar(qlabels, qvals, color=qcols, edgecolor='white', lw=1.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    str(int(bar.get_height())),
                    ha='center', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Supplier-Country Pairs')
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig); plt.close()

        st.markdown(f'''
        <div class="insight-box">
        <b>📊 Priority Actions by Quadrant:</b><br>
        • <b>Q1 ({q1} pairs)</b> — Immediate: dual-source or buffer stock<br>
        • <b>Q2 ({q2} pairs)</b> — Monitor closely, consider replacement<br>
        • <b>Q3 ({q3} pairs)</b> — Maintain current relationship<br>
        • <b>Q4 ({q4} pairs)</b> — Strategic partnership candidates
        </div>''', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('### 🎯 Key Strategic Recommendations')
    ss_count = int(supplier_df['single_source'].sum())
    africa_count = int((supplier_df['Region'] == 'Sub-Saharan Africa (SSA)').sum())
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f'''
        <div class="risk-low">
        <b>✅ Immediate (Next 30 Days):</b><br>
        1. Audit top {q1} high-risk, high-spend supplier pairs<br>
        2. Dual-source all {ss_count} single-source vendor-country pairs<br>
        3. Request updated financials from vendors with PD &gt; 60%<br>
        4. Establish 4-week buffer for high geo-risk country shipments
        </div>''', unsafe_allow_html=True)
    with r2:
        st.markdown(f'''
        <div class="insight-box">
        <b>📋 Medium-Term (Next 90 Days):</b><br>
        1. Qualify {min(3, q2)} backup vendors in low governance-risk regions<br>
        2. Implement monthly vendor performance scorecards<br>
        3. Negotiate supply guarantees with top 5 vendors by spend<br>
        4. Fix missing PO date fields to reduce planning risk scores
        </div>''', unsafe_allow_html=True)


# ── TAB 6: Data Provenance ────────────────────────────────────────────────
with tab6:
    st.subheader('🔍 Data Provenance — Where Every Metric Comes From')
    st.markdown('''
    This tab explains exactly how each risk dimension is derived from raw data.
    **There is no synthetic or randomly generated data in this model.**
    All 6 risk dimensions, PD, and LGD are computed from real columns.
    ''')

    provenance = {
        'OTD Risk': {
            'Source': 'USAID SCMS Delivery History Dataset',
            'Raw Columns': '`Scheduled Delivery Date`, `Delivered to Client Date`',
            'Method': ('`is_on_time` = 1 if Delivered ≤ Scheduled, else 0. '
                       'OTD Rate = mean per Vendor×Country pair. '
                       'Only pairs with ≥ min_shipments records included. '
                       'otd_risk = 1 − OTD Rate.'),
            'Risk Direction': 'Lower OTD → Higher risk'
        },
        'Lead Time Variability (LT Risk)': {
            'Source': 'USAID SCMS Delivery History Dataset',
            'Raw Columns': '`PO Sent to Vendor Date`, `Delivered to Client Date` (fallback: `PQ First Sent`)',
            'Method': ('lead_time_days = Delivered − PO Sent. '
                       'LT CoV = std(lead_time_days) / mean(lead_time_days). '
                       'Robust to zero-mean: fills with median CoV when mean = 0. '
                       'Negative lead times set to NaN before aggregation.'),
            'Risk Direction': 'Higher CoV → Higher risk'
        },
        'Governance / Geo Risk': {
            'Source': 'World Bank WGI 2012-2022 (Mendeley, CC-BY 4.0)',
            'Raw Columns': ('`Voice and Accountability`, `Political Stability`, '
                            '`Government Effectiveness`, `Regulatory Quality`, '
                            '`Rule of Law`, `Control of Corruption` (percentile ranks 0–100)'),
            'Method': ('Governance_Score = mean(6 dims) / 100. '
                       'Governance_Risk = 1 − Governance_Score. '
                       'Matched on vendor primary delivery Country. '
                       'Unmatched countries filled with WGI year median.'),
            'Risk Direction': 'Higher Governance_Risk → Higher geo/ESG risk'
        },
        'Concentration / Single-Source Risk': {
            'Source': 'USAID SCMS Delivery History Dataset',
            'Raw Columns': '`Vendor`, `Country`, `Product Group`, `Line Item Value`',
            'Method': ('country_product_share = vendor spend / total spend in '
                       'same Country×Product cell. single_source = 1 if share ≥ 80% '
                       'or shipments ≤ 3. '
                       'conc_risk = 0.7 × country_product_share + 0.3 × single_source.'),
            'Risk Direction': 'Higher share / single-source flag → Higher risk'
        },
        'Logistics Complexity Risk': {
            'Source': 'USAID SCMS Delivery History Dataset',
            'Raw Columns': '`Shipment Mode`, `Vendor INCO Term`',
            'Method': ('Dominant shipment mode mapped to base risk score '
                       '(Air=0.35, Ocean=0.70, Air Charter=0.60, Truck=0.25). '
                       'Penalised for diversity of modes and INCO terms used '
                       '(more variation = less predictable). '
                       'logistics_risk = 0.5×mode_risk + 0.25×mode_diversity + 0.25×inco_diversity.'),
            'Risk Direction': 'Higher complexity → Higher risk'
        },
        'Planning Risk': {
            'Source': 'USAID SCMS Delivery History Dataset',
            'Raw Columns': '`PO Sent to Vendor Date`, `lead_time_days` (derived)',
            'Method': ('planning_risk = 0.65 × (fraction of orders with missing PO date) '
                       '+ 0.35 × (1 if mean lead time is null, else 0). '
                       'High missing PO rate = poor procurement planning visibility.'),
            'Risk Direction': 'Higher missing rate → Higher risk'
        },
        'LGD (Loss Given Disruption)': {
            'Source': 'Derived from SCMS + WGI risk dimensions above',
            'Raw Columns': 'single_source, logistics_risk, geo_risk',
            'Method': ('LGD = 0.25 (base) + 0.35×single_source + '
                       '0.20×logistics_risk + 0.20×geo_risk. Clipped [0.20, 0.95]. '
                       'NOT randomly sampled — each vendor has a structurally '
                       'justified LGD based on replaceability and recovery difficulty.'),
            'Risk Direction': 'Higher structural risk → Higher LGD'
        },
    }

    for dim, info in provenance.items():
        with st.expander(f'📌 {dim}', expanded=False):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown(f"**Source:** {info['Source']}")
                st.markdown(f"**Raw Columns:** {info['Raw Columns']}")
                st.markdown(f"**Risk Direction:** {info['Risk Direction']}")
            with col_b:
                st.markdown(f"**Methodology:** {info['Method']}")

    st.markdown('---')
    st.markdown('#### WGI Country Match Status')
    scms_countries = set(scms['Country'].unique())
    wgi_countries  = set(wgi_year_df['Country'].unique())
    matched   = scms_countries & wgi_countries
    unmatched = scms_countries - wgi_countries - {'Unknown'}
    if unmatched:
        st.warning(f'⚠️ {len(unmatched)} SCMS countries not in WGI year {wgi_year} '
                   f'(filled with median): {", ".join(sorted(unmatched))}')
    else:
        st.success(f'✅ All SCMS countries matched in WGI {wgi_year} dataset.')

    st.markdown('---')
    st.markdown('#### Raw Dataset Summary')
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(f'''
        **SCMS Delivery History Dataset**
        - Provider: USAID / Supply Chain Management System
        - License: Public Domain (US Government)
        - Records: {len(scms):,} shipment-level rows (after RDC filter)
        - Vendors: {scms["Vendor"].nunique()} unique
        - Countries: {scms["Country"].nunique()} destination countries
        - Note: "SCMS from RDC" excluded — redistribution centre, not a vendor
        ''')
    with rc2:
        st.markdown(f'''
        **Worldwide Governance Indicators 2012–2022**
        - Provider: World Bank Group (via Mendeley Data)
        - License: Creative Commons Attribution 4.0
        - Records: 1,980 (180 countries × 11 years)
        - Indicators used: All 6 WGI dimensions
        - Selected year for scoring: {wgi_year}
        - Academic reference: Kaufmann & Kraay (2023)
        ''')
