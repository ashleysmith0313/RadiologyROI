
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import matplotlib.ticker as mtick

st.set_page_config(page_title="Radiology Locums ROI (Editable)", layout="wide")

# Optional logo
logo_path = Path("logo.png")
if logo_path.exists():
    st.image(str(logo_path), width=220)

st.title("Radiology Locums ROI — Logic-First, Fully Editable")
st.caption("Everything is editable. Use presets, upload a JSON to prefill, or edit assumptions live.")

def init_default_rates():
    return pd.DataFrame({
        "modality_key": ["xray","ct_nocon","ct_con","mri_nocon","mri_con","ultrasound","pet"],
        "description": ["X-Ray","CT (no contrast)","CT (with contrast)","MRI (no contrast)","MRI (with contrast)","Ultrasound","PET"],
        "apc_example": ["","","","","","",""],
        "medicare_rate_usd": [0,0,0,0,0,0,0],
        "commercial_multiplier": [1.0]*7,
        "medicaid_multiplier": [1.0]*7,
        "pct_with_contrast": [0.0,0.0,1.0,0.0,1.0,0.0,1.0],
        "pro_share": [0.2,0.2,0.2,0.25,0.25,0.2,0.25],
    })

def blended_allowed(row, mix):
    base = row.get("medicare_rate_usd", 0.0)
    allow_medicare = base
    allow_commercial = base * row.get("commercial_multiplier", 1.0)
    allow_medicaid = base * row.get("medicaid_multiplier", 1.0)
    return (
        mix["commercial"] * allow_commercial +
        mix["medicaid"] * allow_medicaid +
        mix["medicare"] * allow_medicare
    )

def modality_rate(rates, modality_key_base):
    if modality_key_base in ["ct","mri"]:
        no_key = f"{modality_key_base}_nocon"
        con_key = f"{modality_key_base}_con"
        try:
            no_rate = rates.loc[rates["modality_key"]==no_key, "avg_allowed"].values[0]
            con_rate = rates.loc[rates["modality_key"]==con_key, "avg_allowed"].values[0]
            pct_con = rates.loc[rates["modality_key"]==con_key, "pct_with_contrast"].values[0]
        except Exception:
            return 0.0
        return (1-pct_con)*no_rate + pct_con*con_rate
    else:
        try:
            return rates.loc[rates["modality_key"]==modality_key_base, "avg_allowed"].values[0]
        except Exception:
            return 0.0

def safe_num(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# Presets/config
st.sidebar.header("Start here")
preset_choice = st.sidebar.selectbox("Load a preset", ["None", "Baseline (placeholders)", "High Medicaid (placeholders)"])
preset_map = {}
if Path("presets/baseline.json").exists():
    preset_map["Baseline (placeholders)"] = json.loads(Path("presets/baseline.json").read_text())
if Path("presets/high_medicaid.json").exists():
    preset_map["High Medicaid (placeholders)"] = json.loads(Path("presets/high_medicaid.json").read_text())

if preset_choice != "None" and preset_choice in preset_map:
    st.session_state["cfg"] = preset_map[preset_choice]
elif "cfg" not in st.session_state:
    st.session_state["cfg"] = {}

cfg = st.session_state["cfg"]
uploaded_config = st.sidebar.file_uploader("Or load JSON config", type=["json"])
if uploaded_config is not None:
    try:
        cfg = json.loads(uploaded_config.read().decode("utf-8"))
        st.session_state["cfg"] = cfg
        st.sidebar.success("Config loaded.")
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")

def cfg_get(path, default):
    node = cfg
    for key in path.split("."):
        node = node.get(key, {} if isinstance(node, dict) else None)
        if node is None:
            return default
    return node if node != {} else default

with st.sidebar:
    st.header("Volume drivers")
    ed_visits = st.number_input("Annual ED visits", value=int(cfg_get("volume.ed_visits", 62000)), min_value=0, step=100)
    util_xray = st.number_input("ED X-ray per 100", value=safe_num(cfg_get("volume.util_xray", 48.0)), step=0.5)
    util_ct   = st.number_input("ED CT per 100",   value=safe_num(cfg_get("volume.util_ct", 29.0)), step=0.5)
    util_us   = st.number_input("ED Ultrasound per 100", value=safe_num(cfg_get("volume.util_us", 10.0)), step=0.5)
    util_mri  = st.number_input("ED MRI per 100",  value=safe_num(cfg_get("volume.util_mri", 2.5)), step=0.1)
    util_pet  = st.number_input("ED PET per 100",  value=safe_num(cfg_get("volume.util_pet", 0.2)), step=0.1)

    st.subheader("Non-ED multipliers")
    mult_xray = st.number_input("X-ray multiplier", value=safe_num(cfg_get("volume.mult_xray", 1.8)), step=0.1, min_value=0.0)
    mult_ct   = st.number_input("CT multiplier",    value=safe_num(cfg_get("volume.mult_ct", 2.2)), step=0.1, min_value=0.0)
    mult_us   = st.number_input("Ultrasound multiplier", value=safe_num(cfg_get("volume.mult_us", 2.0)), step=0.1, min_value=0.0)
    mult_mri  = st.number_input("MRI multiplier",   value=safe_num(cfg_get("volume.mult_mri", 2.5)), step=0.1, min_value=0.0)
    mult_pet  = st.number_input("PET multiplier",   value=safe_num(cfg_get("volume.mult_pet", 1.6)), step=0.1, min_value=0.0)

    st.header("Program + Finance")
    capture_wo = st.slider("% studies completed without locums", 0, 100, int(cfg_get("finance.capture_wo", 70)), 1)
    hospital_captures = st.selectbox("Revenue captured by hospital", ["Technical only", "Professional + Technical"], index=0 if cfg_get("finance.hospital_captures", "Technical only")=="Technical only" else 1)
    bad_debt = st.slider("Bad debt/denials %", 0, 30, int(cfg_get("finance.bad_debt", 5)), 1)

    st.subheader("Payer mix (%) — editable")
    pay_commercial = st.number_input("Commercial/Other %", value=int(cfg_get("payer.commercial", 71)), min_value=0, max_value=100, step=1)
    pay_medicaid   = st.number_input("Medicaid %", value=int(cfg_get("payer.medicaid", 15)), min_value=0, max_value=100, step=1)
    pay_medicare   = max(0, 100 - pay_commercial - pay_medicaid)
    st.text(f"Medicare (auto): {pay_medicare}%")

    st.header("Locums program")
    locum_total_cost = st.number_input("Total locums spend ($)", value=safe_num(cfg_get("locums.total_cost", 6000000.0)), step=10000.0, min_value=0.0, format="%.2f")
    reads_per_locum_day = st.number_input("Reads per locum FTE-day", value=safe_num(cfg_get("locums.reads_per_day", 70.0)), step=1.0, min_value=0.0)
    locum_fte_days = st.number_input("Total locum FTE-days (period)", value=safe_num(cfg_get("locums.fte_days", 2000.0)), step=10.0, min_value=0.0)

st.subheader("Rate table (fully editable)")
if "rates_df" not in st.session_state:
    rates_cfg = cfg_get("rates.table", None)
    st.session_state["rates_df"] = pd.DataFrame(rates_cfg) if rates_cfg else init_default_rates()

rates_df = st.data_editor(
    st.session_state["rates_df"],
    num_rows="dynamic",
    use_container_width=True,
    key="rates_editor"
)

# Upload/Download rates CSV
c1, c2 = st.columns(2)
with c1:
    uploaded_rates_csv = st.file_uploader("Upload rates CSV", type=["csv"])
    if uploaded_rates_csv is not None:
        try:
            st.session_state["rates_df"] = pd.read_csv(uploaded_rates_csv)
            rates_df = st.session_state["rates_df"]
            st.success("Rates CSV loaded.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
with c2:
    csv_bytes = rates_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download current rates CSV", data=csv_bytes, file_name="radiology_rates.csv", mime="text/csv")

# Build payer-adjusted average allowed amounts
payer_mix = {
    "commercial": pay_commercial/100.0,
    "medicaid":   pay_medicaid/100.0,
    "medicare":   pay_medicare/100.0,
}
rates = rates_df.copy()
rates["avg_allowed"] = rates.apply(lambda r: blended_allowed(r, payer_mix), axis=1)

# Derive modality rates with contrast mix
rate_xray = modality_rate(rates, "xray")
rate_ct   = modality_rate(rates, "ct")
rate_us   = modality_rate(rates, "ultrasound")
rate_mri  = modality_rate(rates, "mri")
rate_pet  = modality_rate(rates, "pet")

# Adjust for technical-only capture if selected
if hospital_captures == "Technical only":
    def get_pro(k):
        v = rates.loc[rates["modality_key"]==k,"pro_share"]
        return float(v.values[0]) if len(v)>0 else 0.0
    pro_shares = [get_pro("xray"), get_pro("ct_con"), get_pro("mri_con"), get_pro("ultrasound"), get_pro("pet")]
    tech_factor = 1 - np.array(pro_shares)
    rate_xray *= tech_factor[0]
    rate_ct   *= tech_factor[1]
    rate_mri  *= tech_factor[2]
    rate_us   *= tech_factor[3]
    rate_pet  *= tech_factor[4]

# Compute volumes from ED anchor
ed_factor = ed_visits / 100.0
ed_xray = util_xray * ed_factor
ed_ct   = util_ct   * ed_factor
ed_us   = util_us   * ed_factor
ed_mri  = util_mri  * ed_factor
ed_pet  = util_pet  * ed_factor

vol_xray = ed_xray * mult_xray
vol_ct   = ed_ct   * mult_ct
vol_us   = ed_us   * mult_us
vol_mri  = ed_mri  * mult_mri
vol_pet  = ed_pet  * mult_pet

# Incremental studies and capacity
studies_total = vol_xray + vol_ct + vol_us + vol_mri + vol_pet
studies_wo = studies_total * (capture_wo/100.0)
incremental_studies_raw = studies_total - studies_wo

locum_capacity_reads = reads_per_locum_day * locum_fte_days
scale = 1.0
capacity_note = "Capacity OK"
if locum_capacity_reads < incremental_studies_raw and incremental_studies_raw > 0:
    scale = locum_capacity_reads / incremental_studies_raw
    incremental_studies = locum_capacity_reads
    capacity_note = "Capacity-limited; incremental reads scaled to locums capacity."
else:
    incremental_studies = incremental_studies_raw

# Revenue
gross_rev = (
    vol_xray*rate_xray + vol_ct*rate_ct + vol_us*rate_us + vol_mri*rate_mri + vol_pet*rate_pet
)
incremental_rev = gross_rev * (1 - capture_wo/100.0) * scale
net_incremental_rev = incremental_rev * (1 - bad_debt/100.0)

# ROI
net_gain = net_incremental_rev - locum_total_cost
roi_multiple = (net_incremental_rev/locum_total_cost) if locum_total_cost>0 else np.nan

# KPIs
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Incremental reads", int(round(incremental_studies)))
with c2:
    st.metric("Net incremental revenue ($)", f"{net_incremental_rev:,.0f}")
with c3:
    st.metric("Benefit-cost (x)", f"{roi_multiple:,.2f}" if not np.isnan(roi_multiple) else "—")
st.caption(capacity_note)

# ----------------------
# Improved Waterfall
# ----------------------
st.subheader("Waterfall")
gross_before_bd = incremental_rev / (1 - bad_debt/100.0) if (1 - bad_debt/100.0)>0 else 0
steps = [gross_before_bd, -(gross_before_bd - incremental_rev), -locum_total_cost]
labels = ["Gross incremental revenue", "Bad debt/denials", "Locums cost"]
final_label = "Net gain"
final_value = net_gain

cum = [0]
for v in steps:
    cum.append(cum[-1] + v)

fig = plt.figure(figsize=(9,5))
ax = plt.gca()

for i, v in enumerate(steps):
    if v >= 0:
        ax.bar(i, v, bottom=cum[i])
    else:
        ax.bar(i, v, bottom=cum[i] + v)

ax.bar(len(steps), final_value, bottom=0)

ax.set_xticks(range(len(steps)+1))
ax.set_xticklabels(labels + [final_label], rotation=20, ha='right')

ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
ax.grid(axis='y', linestyle='--', alpha=0.6)

def add_label(x, height, y_base):
    ypos = y_base + (height if height>=0 else 0)
    ax.text(x, ypos, f"{height:,.0f}", ha='center', va='bottom', fontsize=9)

for i, v in enumerate(steps):
    y_base = cum[i] if v>=0 else cum[i] + v
    add_label(i, v, y_base)

add_label(len(steps), final_value, 0)

ax.set_ylabel("USD")
plt.tight_layout()
st.pyplot(fig)

# Detail table
st.subheader("Detail: volumes and rates")
detail_df = pd.DataFrame({
    "Modality": ["X-Ray","CT","Ultrasound","MRI","PET"],
    "Annual studies": [vol_xray, vol_ct, vol_us, vol_mri, vol_pet],
    "Avg allowed amount ($)": [rate_xray, rate_ct, rate_us, rate_mri, rate_pet],
    "Annual revenue ($)": [
        vol_xray*rate_xray, vol_ct*rate_ct, vol_us*rate_us, vol_mri*rate_mri, vol_pet*rate_pet
    ]
}).round(2)
st.dataframe(detail_df, use_container_width=True)

# Save/Load JSON
st.divider()
st.subheader("Save/Load everything as JSON")
save_dict = {
    "volume": {
        "ed_visits": ed_visits,
        "util_xray": util_xray,
        "util_ct": util_ct,
        "util_us": util_us,
        "util_mri": util_mri,
        "util_pet": util_pet,
        "mult_xray": mult_xray,
        "mult_ct": mult_ct,
        "mult_us": mult_us,
        "mult_mri": mult_mri,
        "mult_pet": mult_pet,
    },
    "finance": {
        "capture_wo": capture_wo,
        "hospital_captures": hospital_captures,
        "bad_debt": bad_debt,
    },
    "payer": {
        "commercial": pay_commercial,
        "medicaid": pay_medicaid,
    },
    "locums": {
        "total_cost": locum_total_cost,
        "reads_per_day": reads_per_locum_day,
        "fte_days": locum_fte_days,
    },
    "rates": {
        "table": rates_df.to_dict(orient="list")
    }
}
json_bytes = json.dumps(save_dict, indent=2).encode("utf-8")
st.download_button("Download JSON config", data=json_bytes, file_name="radiology_roi_config.json", mime="application/json")

# PDF summary
st.divider()
st.subheader("Create a one-page PDF summary")
if st.button("Generate PDF"):
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter
    y = height - 1*inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, y, "Radiology Locums ROI — Summary"); y -= 0.3*inch
    c.setFont("Helvetica-Bold", 12); c.drawString(1*inch, y, "Key Inputs"); y -= 0.2*inch
    c.setFont("Helvetica", 10)
    lines = [
        f"ED visits: {ed_visits:,}  |  Capture without locums: {capture_wo}%  |  Bad debt: {bad_debt}%",
        f"Payer mix: Comm {int(100* payer_mix['commercial'])}%, Medicaid {int(100* payer_mix['medicaid'])}%, Medicare {int(100* payer_mix['medicare'])}%",
        f"Locums: ${locum_total_cost:,.0f} spend, {locum_fte_days:,.0f} FTE-days @ {reads_per_locum_day:,.0f} reads/day"
    ]
    for line in lines:
        c.drawString(1*inch, y, line); y -= 0.18*inch

    y -= 0.1*inch
    c.setFont("Helvetica-Bold", 12); c.drawString(1*inch, y, "Results"); y -= 0.2*inch
    c.setFont("Helvetica", 10)
    c.drawString(1*inch, y, f"Incremental reads: {int(round(incremental_studies)):,}"); y -= 0.18*inch
    c.drawString(1*inch, y, f"Net incremental revenue: ${net_incremental_rev:,.0f}"); y -= 0.18*inch
    if not np.isnan(roi_multiple):
        c.drawString(1*inch, y, f"Benefit-cost (x): {roi_multiple:.2f}"); y -= 0.18*inch
    c.drawString(1*inch, y, f"Net gain: ${net_gain:,.0f}"); y -= 0.18*inch

    y -= 0.2*inch
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(1*inch, y, "Note: All inputs are user-editable; this summary is illustrative only.")
    c.showPage()
    c.save()
    pdf_buffer.seek(0)
    st.download_button("Download PDF summary", data=pdf_buffer.getvalue(), file_name="radiology_locums_roi_summary.pdf", mime="application/pdf")
