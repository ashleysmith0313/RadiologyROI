
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
import PyPDF2

def _normalize_rates_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Make the rates CSV forgiving:
    - trims and lowercases headers
    - maps common header variants -> required names
    - coerces numeric fields
    - fills any missing required columns with defaults
    Required columns:
      modality_key, description, apc_example, medicare_rate_usd,
      commercial_multiplier, medicaid_multiplier, pct_with_contrast, pro_share
    """
    req_cols = ["modality_key","description","apc_example","medicare_rate_usd",
                "commercial_multiplier","medicaid_multiplier","pct_with_contrast","pro_share"]

    # Header normalize
    df = df_raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Common aliases
    aliases = {
        "modality": "modality_key",
        "modalityname": "modality_key",
        "modality_key": "modality_key",
        "desc": "description",
        "apc": "apc_example",
        "apc_levels": "apc_example",
        "medicare": "medicare_rate_usd",
        "medicare_rate": "medicare_rate_usd",
        "medicare_rate_$": "medicare_rate_usd",
        "medicare_rate_usd": "medicare_rate_usd",
        "commercial_mult": "commercial_multiplier",
        "commercial_multiplier": "commercial_multiplier",
        "comm_multiplier": "commercial_multiplier",
        "comm_mult": "commercial_multiplier",
        "medicaid_mult": "medicaid_multiplier",
        "medicaid_multiplier": "medicaid_multiplier",
        "pct_contrast": "pct_with_contrast",
        "percent_with_contrast": "pct_with_contrast",
        "with_contrast_pct": "pct_with_contrast",
        "pct_with_contrast": "pct_with_contrast",
        "professional_share": "pro_share",
        "proportion_professional": "pro_share",
        "pro_pct": "pro_share",
        "pro_share": "pro_share",
    }

    # Map aliases
    mapped = {}
    for c in df.columns:
        key = aliases.get(c, c)
        mapped[key] = df[c]
    df = pd.DataFrame(mapped)

    # Ensure required columns exist
    for col in req_cols:
        if col not in df.columns:
            if col in ["description","apc_example"]:
                df[col] = ""
            elif col == "modality_key":
                # try to derive from description if present
                if "description" in df.columns:
                    df[col] = df["description"].str.lower().str.extract(r"(x[- ]?ray|ct.*no|ct.*with|mri.*no|mri.*with|ultra|pet)", expand=False).fillna("")
                else:
                    df[col] = ""
            elif col in ["pct_with_contrast","pro_share"]:
                df[col] = 0.0
            else:
                df[col] = 0.0

    # Coerce numeric
    for col in ["medicare_rate_usd","commercial_multiplier","medicaid_multiplier","pct_with_contrast","pro_share"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Keep only required columns (ordered)
    df = df[req_cols]

    return df

st.set_page_config(page_title="Radiology Locums ROI", layout="wide")

# ---------------------
# Simple top navigation
# ---------------------
page = st.sidebar.radio("Navigate", ["Home","Inputs","Rates","Results","Export"])

# persistent state
if "uploaded" not in st.session_state:
    st.session_state["uploaded"] = {}
if "config" not in st.session_state:
    st.session_state["config"] = {}

# ---------------------
# Helper functions
# ---------------------
def load_excel_preview(upload):
    try:
        df = pd.read_excel(upload)
        return df.head(200)
    except Exception as e:
        return pd.DataFrame({"Error":[str(e)]})

def read_pdf_quick_stats(upload):
    try:
        reader = PyPDF2.PdfReader(upload)
        pages = len(reader.pages)
        sample = ""
        if pages>0:
            text = reader.pages[0].extract_text() or ""
            sample = text[:500]
        return {"pages": pages, "sample_text": sample}
    except Exception as e:
        return {"error": str(e)}

def currency(x):
    try:
        return f"${x:,.0f}"
    except:
        return str(x)

# ---------------------
# Landing Page
# ---------------------
if page == "Home":
    st.title("Radiology Locums ROI")
    st.caption("A simple, client-friendly way to show how locums reads convert into incremental studies, revenue, and ROI.")

    c1,c2 = st.columns([1,1])
    with c1:
        st.header("Step 1 — Upload your files")
        st.write("Use any of these (optional, you can also type values later):")
        rates_file = st.file_uploader("Rates CSV (e.g., radiology_rates.csv)", type=["csv"], key="rates_csv")
        excel_file = st.file_uploader("Forecast/volumes Excel (e.g., Queen's Monthly Forecast)", type=["xlsx","xls"], key="xlsx")
        pdf_files = st.file_uploader("Productivity PDFs (drag in one or more)", type=["pdf"], accept_multiple_files=True, key="pdfs")

        # Save into session for other pages
        if rates_file: st.session_state["uploaded"]["rates_csv"] = rates_file
        if excel_file: st.session_state["uploaded"]["excel"] = excel_file
        if pdf_files:  st.session_state["uploaded"]["pdfs"]  = pdf_files

        if excel_file:
            st.write("**Excel quick preview:** (first sheet, first 200 rows)")
            st.dataframe(load_excel_preview(excel_file), use_container_width=True)

        if pdf_files:
            st.write("**PDF quick stats:**")
            for f in pdf_files:
                stats = read_pdf_quick_stats(f)
                st.markdown(f"- `{f.name}` — pages: **{stats.get('pages','?')}**")
                if "sample_text" in stats and stats["sample_text"]:
                    with st.expander(f"Sample text from {f.name}"):
                        st.text(stats["sample_text"])

    with c2:
        st.header("Step 2 — Review assumptions")
        st.markdown("""
        - **ED anchor** → converts visits into imaging using per-100 norms  
        - **Non-ED multipliers** → add OP/IP imaging  
        - **Payer mix** + **rates** → price each study  
        - **Without-locums %** + **locums capacity** → true incremental reads  
        - **Bad-debt/denials** → net dollars  
        """)
        st.header("Step 3 — See the story")
        st.markdown("""
        - KPIs: incremental reads, net incremental revenue, ROI  
        - **Waterfall**: Gross → Denials → Locums cost → **Net gain**  
        - **Detail table**: volumes × allowed amount by modality  
        - **Export**: PDF summary + JSON of your scenario  
        """)
        st.info("Use the left **Navigate** menu to go to **Inputs**, **Rates**, **Results**, and **Export**.")

# ---------------------
# Inputs page (same fields, cleaner layout)
# ---------------------
if page == "Inputs":
    st.header("Inputs — volumes, finance, payer mix, locums capacity")

    with st.expander("Volume drivers (ED → imaging → total system)", expanded=True):
        col = st.columns(5)
        ed_visits = col[0].number_input("Annual ED visits", value=62000, min_value=0, step=100)
        x_per100 = col[1].number_input("X-ray /100 ED", value=48.0, step=0.5)
        ct_per100 = col[2].number_input("CT /100 ED", value=29.0, step=0.5)
        us_per100 = col[3].number_input("US /100 ED", value=10.0, step=0.5)
        mri_per100 = col[4].number_input("MRI /100 ED", value=2.5, step=0.1)
        pet_per100 = st.number_input("PET /100 ED", value=0.2, step=0.1)
        st.caption("Use norms from ACEP/ED Benchmarking or your own.")

        colm = st.columns(5)
        mult_x = colm[0].number_input("X-ray multiplier", value=1.8, step=0.1, min_value=0.0)
        mult_ct = colm[1].number_input("CT multiplier", value=2.2, step=0.1, min_value=0.0)
        mult_us = colm[2].number_input("US multiplier", value=2.0, step=0.1, min_value=0.0)
        mult_mri = colm[3].number_input("MRI multiplier", value=2.5, step=0.1, min_value=0.0)
        mult_pet = colm[4].number_input("PET multiplier", value=1.6, step=0.1, min_value=0.0)

    with st.expander("Program & finance", expanded=True):
        colf = st.columns(4)
        capture_wo = colf[0].slider("% studies done without locums", 0, 100, 70, 1)
        bad_debt = colf[1].slider("Bad debt/denials %", 0, 30, 5, 1)
        hospital_captures = colf[2].selectbox("Hospital captures", ["Technical only", "Professional + Technical"])
        pay_comm = colf[3].number_input("Commercial %", value=71, min_value=0, max_value=100, step=1)  # value, min, max
        colf2 = st.columns(2)
        pay_mcaid = colf2[0].number_input("Medicaid %", value=15, min_value=0, max_value=100, step=1)
        pay_mcare = max(0, 100 - pay_comm - pay_mcaid)
        colf2[1].metric("Medicare (auto)", f"{pay_mcare}%")

    with st.expander("Locums capacity", expanded=True):
        coll = st.columns(3)
        locum_cost = coll[0].number_input("Total locums spend ($)", value=6000000.0, step=10000.0, format="%.2f", min_value=0.0)
        reads_day = coll[1].number_input("Reads per FTE-day", value=70.0, step=1.0, min_value=0.0)
        fte_days = coll[2].number_input("Total FTE-days", value=2000.0, step=10.0, min_value=0.0)

    # Save to session for other pages
    st.session_state["inputs"] = dict(
        ed_visits=ed_visits, x_per100=x_per100, ct_per100=ct_per100, us_per100=us_per100,
        mri_per100=mri_per100, pet_per100=pet_per100, mult_x=mult_x, mult_ct=mult_ct,
        mult_us=mult_us, mult_mri=mult_mri, mult_pet=mult_pet, capture_wo=capture_wo,
        bad_debt=bad_debt, hospital_captures=hospital_captures, pay_comm=pay_comm,
        pay_mcaid=pay_mcaid, pay_mcare=pay_mcare, locum_cost=locum_cost,
        reads_day=reads_day, fte_days=fte_days
    )
    st.success("Inputs captured. Move to **Rates** to edit pricing.")

# ---------------------
# Rates page (grid editor + optional CSV upload)
# ---------------------
if page == "Rates":
    st.header("Rates — allowed amounts and pro share")
    st.write("Upload a CSV (or edit inline). We’ll blend with payer mix and contrast % for CT/MRI.")
    
rates_upload = st.file_uploader("Upload rates CSV", type=["csv"], key="rates_upload_page")
df = None
if rates_upload is not None:
    df = _normalize_rates_df(pd.read_csv(rates_upload))
elif "rates_csv" in st.session_state["uploaded"]:
    try:
        df = _normalize_rates_df(pd.read_csv(st.session_state["uploaded"]["rates_csv"]))
    except Exception:
        df = None
if df is None:
    df = pd.DataFrame({
        "modality_key": ["xray","ct_nocon","ct_con","mri_nocon","mri_con","ultrasound","pet"],
        "description": ["X-Ray","CT (no contrast)","CT (with contrast)","MRI (no contrast)","MRI (with contrast)","Ultrasound","PET"],
        "apc_example": ["5521-5524","5521-5524","5571-5573","5523-5524","5571-5573","Various","Cardiac PET APC"],
        "medicare_rate_usd": [88, 250, 550, 450, 790, 220, 1950],
        "commercial_multiplier": [2.0]*7,
        "medicaid_multiplier": [0.8]*7,
        "pct_with_contrast": [0.0,0.20,0.80,0.15,0.85,0.0,1.0],
        "pro_share": [0.2,0.2,0.2,0.25,0.25,0.2,0.25],
    })

    if rates_upload:
        df = pd.read_csv(rates_upload)
    else:
        # Use uploaded CSV from Home if available
        df = None
        if "rates_csv" in st.session_state["uploaded"]:
            try:
                df = pd.read_csv(st.session_state["uploaded"]["rates_csv"])
            except Exception:
                pass
        if df is None:
            df = pd.DataFrame({
                "modality_key": ["xray","ct_nocon","ct_con","mri_nocon","mri_con","ultrasound","pet"],
                "description": ["X-Ray","CT (no contrast)","CT (with contrast)","MRI (no contrast)","MRI (with contrast)","Ultrasound","PET"],
                "apc_example": ["5521-5524","5521-5524","5571-5573","5523-5524","5571-5573","Various","Cardiac PET APC"],
                "medicare_rate_usd": [88, 250, 550, 450, 790, 220, 1950],
                "commercial_multiplier": [2.0]*7,
                "medicaid_multiplier": [0.8]*7,
                "pct_with_contrast": [0.0,0.20,0.80,0.15,0.85,0.0,1.0],
                "pro_share": [0.2,0.2,0.2,0.25,0.25,0.2,0.25],
            })

    edited = st.data_editor(df, use_container_width=True, num_rows="dynamic", key="rates_editor_home")
    st.session_state["rates"] = edited
    st.info("Tip: Click **Results** when you’re done editing.")

# ---------------------
# Results page (clean KPIs + improved waterfall + detail table)
# ---------------------
if page == "Results":
    st.header("Results — what did locums add?")
    if "inputs" not in st.session_state or "rates" not in st.session_state:
        st.warning("Go to **Inputs** and **Rates** first.")
    else:
        I = st.session_state["inputs"]
        R = st.session_state["rates"]

        # Payer mix
        payer_mix = {
            "commercial": I["pay_comm"]/100.0,
            "medicaid": I["pay_mcaid"]/100.0,
            "medicare": I["pay_mcare"]/100.0,
        }

        # Blend allowed amounts
        rates = R.copy()
        def blend(row):
            base = row.get("medicare_rate_usd", 0.0)
            return (
                payer_mix["commercial"] * (base*row.get("commercial_multiplier",1.0)) +
                payer_mix["medicaid"] * (base*row.get("medicaid_multiplier",1.0)) +
                payer_mix["medicare"] * base
            )
        rates["avg_allowed"] = rates.apply(blend, axis=1)

        # Modality rates w/ contrast blending
        def modality_rate(modality_key_base):
            if modality_key_base in ["ct","mri"]:
                no_key = f"{modality_key_base}_nocon"
                con_key = f"{modality_key_base}_con"
                no_rate = rates.loc[rates["modality_key"]==no_key,"avg_allowed"].values[0]
                con_rate = rates.loc[rates["modality_key"]==con_key,"avg_allowed"].values[0]
                pct_con = rates.loc[rates["modality_key"]==con_key,"pct_with_contrast"].values[0]
                return (1-pct_con)*no_rate + pct_con*con_rate
            else:
                return rates.loc[rates["modality_key"]==modality_key_base,"avg_allowed"].values[0]

        rt_x = modality_rate("xray")
        rt_ct = modality_rate("ct")
        rt_us = modality_rate("ultrasound")
        rt_mri = modality_rate("mri")
        rt_pet = modality_rate("pet")

        if I["hospital_captures"] == "Technical only":
            def pro(k):
                v = rates.loc[rates["modality_key"]==k,"pro_share"]
                return float(v.values[0]) if len(v)>0 else 0.0
            factors = [1-pro("xray"), 1-pro("ct_con"), 1-pro("mri_con"), 1-pro("ultrasound"), 1-pro("pet")]
            rt_x *= factors[0]; rt_ct *= factors[1]; rt_mri *= factors[2]; rt_us *= factors[3]; rt_pet *= factors[4]

        # Volumes
        ed_factor = I["ed_visits"]/100.0
        ed_x = I["x_per100"]*ed_factor; ed_ct = I["ct_per100"]*ed_factor
        ed_us = I["us_per100"]*ed_factor; ed_mri = I["mri_per100"]*ed_factor; ed_pet = I["pet_per100"]*ed_factor
        v_x = ed_x*I["mult_x"]; v_ct = ed_ct*I["mult_ct"]; v_us = ed_us*I["mult_us"]; v_mri = ed_mri*I["mult_mri"]; v_pet = ed_pet*I["mult_pet"]

        # Incremental + capacity
        total = v_x+v_ct+v_us+v_mri+v_pet
        wo = total*(I["capture_wo"]/100.0)
        inc_raw = total-wo
        capacity = I["reads_day"]*I["fte_days"]
        scale = min(1.0, capacity/inc_raw) if inc_raw>0 else 0.0
        inc_reads = inc_raw if scale==1.0 else capacity

        gross_rev = v_x*rt_x + v_ct*rt_ct + v_us*rt_us + v_mri*rt_mri + v_pet*rt_pet
        inc_rev = gross_rev*(1-I["capture_wo"]/100.0)*scale
        net_inc_rev = inc_rev*(1-I["bad_debt"]/100.0)

        net_gain = net_inc_rev - I["locum_cost"]
        roi_multiple = (net_inc_rev/I["locum_cost"]) if I["locum_cost"]>0 else np.nan

        k1,k2,k3 = st.columns(3)
        k1.metric("Incremental reads", int(round(inc_reads)))
        k2.metric("Net incremental revenue", currency(net_inc_rev))
        k3.metric("Benefit-cost (x)", f"{roi_multiple:.2f}" if not np.isnan(roi_multiple) else "—")

        # Waterfall
        st.subheader("Waterfall")
        gross_before_bd = inc_rev / (1 - I["bad_debt"]/100.0) if (1 - I["bad_debt"]/100.0) > 0 else 0
        steps = [gross_before_bd, -(gross_before_bd - inc_rev), -I["locum_cost"]]
        labels = ["Gross incremental revenue", "Bad debt/denials", "Locums cost"]
        final_value = net_gain

        cum = [0]
        for v in steps: cum.append(cum[-1]+v)

        fig = plt.figure(figsize=(9,5)); ax = plt.gca()
        for i, v in enumerate(steps):
            ax.bar(i, v, bottom=cum[i] if v>=0 else cum[i]+v)
        ax.bar(len(steps), final_value, bottom=0)
        ax.set_xticks(range(len(steps)+1)); ax.set_xticklabels(labels + ["Net gain"], rotation=20, ha='right')
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        def add_label(x, h, b): ax.text(x, b+(h if h>=0 else 0), f"{h:,.0f}", ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(steps): add_label(i, v, cum[i] if v>=0 else cum[i]+v)
        add_label(len(steps), final_value, 0); ax.set_ylabel("USD"); plt.tight_layout()
        st.pyplot(fig)

        # Details
        st.subheader("Detail table")
        detail = pd.DataFrame({
            "Modality":["X-ray","CT","Ultrasound","MRI","PET"],
            "Annual studies":[v_x,v_ct,v_us,v_mri,v_pet],
            "Avg allowed amount ($)":[rt_x,rt_ct,rt_us,rt_mri,rt_pet],
            "Annual revenue ($)":[v_x*rt_x, v_ct*rt_ct, v_us*rt_us, v_mri*rt_mri, v_pet*rt_pet]
        }).round(2)
        st.dataframe(detail, use_container_width=True)

# ---------------------
# Export page
# ---------------------
if page == "Export":
    st.header("Export — PDF summary and JSON")
    if "inputs" not in st.session_state or "rates" not in st.session_state:
        st.warning("Go to **Results** first so we can compute values.")
    else:
        I = st.session_state["inputs"]
        R = st.session_state["rates"]
        # We won't regenerate charts here; just create a simple one-page PDF like before
        from reportlab.pdfgen import canvas
        pdf = BytesIO()
        c = canvas.Canvas(pdf, pagesize=letter); width, height = letter; y = height - 1*inch
        c.setFont("Helvetica-Bold", 16); c.drawString(1*inch, y, "Radiology Locums ROI — Summary"); y -= 0.3*inch
        c.setFont("Helvetica", 10)
        for line in [
            f"ED visits: {I['ed_visits']:,} | Capture without locums: {I['capture_wo']}% | Bad debt: {I['bad_debt']}%",
            f"Payer mix: Comm {I['pay_comm']}% | Medicaid {I['pay_mcaid']}% | Medicare {I['pay_mcare']}%",
            f"Locums: ${I['locum_cost']:,.0f} spend, {I['fte_days']:,.0f} FTE-days @ {I['reads_day']:,.0f} reads/day"
        ]:
            c.drawString(1*inch, y, line); y -= 0.18*inch
        c.setFont("Helvetica-Oblique", 8); y -= 0.2*inch
        c.drawString(1*inch, y, "All inputs editable; figures illustrative."); c.showPage(); c.save(); pdf.seek(0)
        st.download_button("Download PDF summary", data=pdf.getvalue(), file_name="radiology_locums_roi_summary.pdf", mime="application/pdf")

        json_bytes = json.dumps({"inputs": I, "rates": R.to_dict(orient="list")}, indent=2).encode("utf-8")
        st.download_button("Download JSON scenario", data=json_bytes, file_name="radiology_locums_roi_scenario.json", mime="application/json")

