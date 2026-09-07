"""
StatVerify AI

Main Streamlit Application — Multi-Step Guided Workflow Wizard UI
"""

from __future__ import annotations

import pandas as pd

import os
import tempfile

import streamlit as st

from src.models import (
    AnalysisMode,
    AnalysisRequest,
    ColumnRoleMapping,
    InputType,
    StatisticalTest,
    ValueType,
)

from src.parser import UniversalParser
from src.workflow import workflow


# ==========================================================
# StatVerify App Class
# ==========================================================


class StatVerifyApp:
    """
    Programmatic interface to the workflow.

    Used by tests and verification scripts.
    """

    def run(
        self,
        request: AnalysisRequest,
        selected_test=None,
    ) -> dict:
        """
        Execute the workflow with the given request.

        Returns the final workflow state.
        """

        state = {
            "request": request,
            "selected_test": selected_test,
            "parsed_table": None,
            "context": None,
            "dataset": None,
            "test_recommendation": None,
            "statistical_result": None,
            "reported_result": None,
            "verification_result": None,
            "report": None,
            "report_path": None,
            "report_pdf": None,
            "ai_response": None,
            "error": None,
            "group_size_override": None,
        }

        result = workflow.invoke(state)

        return {
            "request": result.get("request"),
            "test_recommendation": result.get("test_recommendation"),
            "statistical_result": result.get("statistical_result"),
            "verification_result": result.get("verification_result"),
            "ai_response": result.get("ai_response"),
            "report": result.get("report"),
            "pdf_path": result.get("report_path"),
            "report_pdf": result.get("report_pdf"),
            "error": result.get("error"),
        }


# ==========================================================
# Streamlit UI
# ==========================================================

if __name__ == "__main__" or hasattr(st, "_is_running_with_streamlit"):

    # ----------------------------------------------------------
    # Page Configuration & Styling
    # ----------------------------------------------------------

    st.set_page_config(
        page_title="StatVerify AI",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Load custom CSS stylesheet
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Initialize wizard step session state
    if "wizard_step" not in st.session_state:
        st.session_state["wizard_step"] = 1

    current_step = st.session_state["wizard_step"]

    # ----------------------------------------------------------
    # Hero Header Section
    # ----------------------------------------------------------

    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-logo-icon">📊</div>
            <div class="hero-title">StatVerify AI</div>
            <div class="hero-subtitle">
                Automated Statistical Analysis, Verification, and AI Explanation Platform
            </div>
            <div class="hero-pills">
                <span class="pill-badge">⚡ CSV / Excel / PDF / Image Ingestion</span>
                <span class="pill-badge">🧬 SciPy & StatsModels Engine</span>
                <span class="pill-badge">🔍 Reported Claim Verification</span>
                <span class="pill-badge">🤖 Gemini Vision & Explanation</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------
    # Stepper Navigation Header Bar
    # ----------------------------------------------------------

    s1_class = "completed" if current_step > 1 else ("active" if current_step == 1 else "")
    s2_class = "completed" if current_step > 2 else ("active" if current_step == 2 else "")
    s3_class = "completed" if current_step > 3 else ("active" if current_step == 3 else "")
    s4_class = "completed" if current_step > 4 else ("active" if current_step == 4 else "")

    l1_class = "completed" if current_step > 1 else ""
    l2_class = "completed" if current_step > 2 else ""
    l3_class = "completed" if current_step > 3 else ""

    st.markdown(
        f"""
        <div class="stepper-container">
            <div class="step-item {s1_class}">
                <div class="step-number">1</div>
                <span>Data Ingestion</span>
            </div>
            <div class="step-line {l1_class}"></div>
            <div class="step-item {s2_class}">
                <div class="step-number">2</div>
                <span>Test Config</span>
            </div>
            <div class="step-line {l2_class}"></div>
            <div class="step-item {s3_class}">
                <div class="step-number">3</div>
                <span>Results & Verification</span>
            </div>
            <div class="step-line {l3_class}"></div>
            <div class="step-item {s4_class}">
                <div class="step-number">4</div>
                <span>AI & PDF Report</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------
    # STEP 1: Data Ingestion & Mode Selection
    # ----------------------------------------------------------

    if current_step == 1:
        st.markdown("<div class='panel-header'>📁 Step 1: Select Analysis Mode & Upload Dataset</div>", unsafe_allow_html=True)

        col_input1, col_input2 = st.columns([1, 1])

        with col_input1:
            mode_choice = st.radio(
                "Select Operating Mode",
                ["Calculation Mode", "Verification Mode"],
                horizontal=True,
                help="Calculation Mode: Computes statistical tests on raw or summary data.\nVerification Mode: Compares reported paper values against exact calculations.",
            )
            st.session_state["mode"] = "Calculation" if mode_choice == "Calculation Mode" else "Verification"

            st.caption("Upload your dataset or choose a pre-loaded sample dataset below:")

            sample_options = {
                "(Select pre-loaded sample dataset...)": None,
                "📊 Demographic Summary CSV (Summary Mode)": "data/demographic_summary.csv",
                "📊 Demographic Summary Excel (Summary Mode)": "data/demographic_summary.xlsx",
                "📄 Demographic Summary PDF (PDF Mode)": "data/demographic_summary.pdf",
                "🧪 Raw Participant Study CSV (Raw Mode)": "data/sample_raw_dataset.csv",
                "🔍 Verification Study CSV (Verification Mode)": "data/verification_raw_study.csv",
            }

            selected_sample_label = st.selectbox(
                "Quick Load Sample Dataset",
                list(sample_options.keys()),
            )
            sample_file_path = sample_options.get(selected_sample_label)

        with col_input2:
            uploaded_files = st.file_uploader(
                "Upload Custom File (CSV, Excel, PDF, Image)",
                type=["csv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg", "tiff"],
                accept_multiple_files=True,
                help="Drag and drop dataset files.",
            )

        # Resolve active file
        file_path = None
        if uploaded_files:
            if len(uploaded_files) > 1:
                selected_file_name = st.selectbox(
                    "Select Active File to Analyze",
                    [f.name for f in uploaded_files],
                )
                active_file = next(f for f in uploaded_files if f.name == selected_file_name)
            else:
                active_file = uploaded_files[0]

            file_id = f"{active_file.name}_{active_file.size}"
            if st.session_state.get("_active_file_id") != file_id or not st.session_state.get("_active_file_path"):
                suffix = os.path.splitext(active_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(active_file.getvalue())
                    file_path = tmp.name

                st.session_state["_active_file_id"] = file_id
                st.session_state["_active_file_path"] = file_path
                st.session_state.pop("_role_mapping", None)
                st.session_state.pop("analysis_result", None)
            else:
                file_path = st.session_state["_active_file_path"]

        elif sample_file_path and os.path.exists(sample_file_path):
            file_path = sample_file_path
            if st.session_state.get("_active_sample_path") != sample_file_path:
                st.session_state["_active_sample_path"] = sample_file_path
                st.session_state.pop("_role_mapping", None)
                st.session_state.pop("analysis_result", None)

        st.session_state["active_file_path"] = file_path

        # Data preview card
        if file_path is not None:
            st.divider()
            try:
                preview_table = UniversalParser().parse(file_path)
                st.session_state["preview_table"] = preview_table
                st.markdown(f"### 📋 Dataset Preview: `{os.path.basename(file_path)}` ({preview_table.dataframe.shape[0]} rows × {preview_table.dataframe.shape[1]} cols)")
                st.dataframe(preview_table.dataframe.head(10), use_container_width=True, hide_index=True)
            except Exception as ex:
                st.error(f"Could not parse file: {ex}")
                st.session_state["preview_table"] = None

            col_next1, col_next2 = st.columns([3, 1])
            with col_next2:
                if st.button("Next: Test Configuration ➡️", use_container_width=True):
                    st.session_state["wizard_step"] = 2
                    st.rerun()
        else:
            st.info("👆 Upload a file or select a sample dataset above to continue.")

    # ----------------------------------------------------------
    # STEP 2: Table Structure & Test Configuration
    # ----------------------------------------------------------

    elif current_step == 2:
        st.markdown("<div class='panel-header'>⚙️ Step 2: Confirm Table Structure & Select Statistical Test</div>", unsafe_allow_html=True)

        file_path = st.session_state.get("active_file_path")
        preview_table = st.session_state.get("preview_table")

        if file_path is None or preview_table is None:
            st.warning("No active dataset file found. Please go back to Step 1.")
            if st.button("⬅️ Back to Step 1"):
                st.session_state["wizard_step"] = 1
                st.rerun()
            st.stop()

        role_mapping = st.session_state.get("_role_mapping")

        # Table structure confirmation
        if preview_table.requires_role_confirmation:
            st.markdown("### 🛠️ Table Role Assignment")
            st.caption("Summary table detected. Assign column roles for exact statistical calculation:")

            columns = list(preview_table.dataframe.columns)
            grouping_options = ["(none)"] + columns
            default_grouping = (
                preview_table.candidate_grouping_column
                if preview_table.candidate_grouping_column in columns
                else "(none)"
            )

            col_r1, col_r2 = st.columns(2)

            with col_r1:
                grouping_choice = st.selectbox(
                    "Variable / grouping column",
                    grouping_options,
                    index=grouping_options.index(default_grouping),
                    key="grouping_select",
                )

                subcat_default = (
                    preview_table.candidate_subcategory_column
                    if preview_table.candidate_subcategory_column in columns
                    else columns[0]
                )

                subcategory_choice = st.selectbox(
                    "Sub-category column",
                    columns,
                    index=columns.index(subcat_default),
                    key="subcat_select",
                )

            with col_r2:
                value_defaults = [
                    c for c in preview_table.candidate_value_columns
                    if c in columns
                ]

                value_columns_choice = st.multiselect(
                    "Group value columns (2 or more)",
                    columns,
                    default=value_defaults,
                    key="values_select",
                )

                value_type_choice = st.radio(
                    "Values represent",
                    ["Counts", "Percentages"],
                    horizontal=True,
                    key="valtype_select",
                )

            group_sizes: dict[str, int] = {}
            if value_type_choice == "Percentages" and value_columns_choice:
                cols_n = st.columns(len(value_columns_choice))
                for idx, col in enumerate(value_columns_choice):
                    with cols_n[idx]:
                        group_sizes[col] = st.number_input(
                            f"Sample N for '{col}'",
                            min_value=1,
                            step=1,
                            value=30,
                            key=f"n_{col}",
                        )

            confirm_disabled = len(value_columns_choice) < 2
            if st.button("Confirm Structure Roles", disabled=confirm_disabled):
                st.session_state["_role_mapping"] = ColumnRoleMapping(
                    grouping_column=(
                        None if grouping_choice == "(none)" else grouping_choice
                    ),
                    sub_category_column=subcategory_choice,
                    group_value_columns=value_columns_choice,
                    value_type=(
                        ValueType.COUNT
                        if value_type_choice == "Counts"
                        else ValueType.PERCENTAGE
                    ),
                    group_sample_sizes=group_sizes,
                )
                role_mapping = st.session_state["_role_mapping"]
                st.success("✅ Structure confirmed!")

            if role_mapping:
                st.info("✅ Table structure validated.")
        else:
            st.success("✅ Tabular dataset structure validated automatically.")

        st.divider()

        # Test selection
        st.markdown("### 🧪 Statistical Test Selection")
        test_options = {
            "Auto (Recommended)": None,
            "Independent t-test": StatisticalTest.T_TEST,
            "Paired t-test": StatisticalTest.PAIRED_T_TEST,
            "One-way ANOVA": StatisticalTest.ANOVA,
            "Mann-Whitney U": StatisticalTest.MANN_WHITNEY,
            "Wilcoxon Signed Rank": StatisticalTest.WILCOXON,
            "Kruskal-Wallis": StatisticalTest.KRUSKAL,
            "Chi-Square": StatisticalTest.CHI_SQUARE,
            "Fisher Exact": StatisticalTest.FISHER_EXACT,
            "Pearson Correlation": StatisticalTest.PEARSON,
            "Spearman Correlation": StatisticalTest.SPEARMAN,
            "Linear Regression": StatisticalTest.LINEAR_REGRESSION,
        }

        selected_test_name = st.selectbox(
            "Select Statistical Test Override",
            list(test_options.keys()),
            help="Auto (Recommended) lets the Decision Engine automatically select the optimal test.",
        )
        selected_test = test_options[selected_test_name]
        st.session_state["selected_test"] = selected_test

        st.divider()

        col_b1, col_b2, col_b3 = st.columns([1, 2, 2])
        with col_b1:
            if st.button("⬅️ Back to Step 1", use_container_width=True):
                st.session_state["wizard_step"] = 1
                st.rerun()

        with col_b3:
            if st.button("🚀 Execute Statistical Analysis & Verification", use_container_width=True):
                if preview_table.requires_role_confirmation and role_mapping is None:
                    st.warning("Please confirm table structure roles before executing.")
                    st.stop()

                suffix = os.path.splitext(file_path)[1]
                input_type_val = (
                    InputType.CSV if suffix.lower() == ".csv"
                    else InputType.EXCEL if suffix.lower() in [".xlsx", ".xls"]
                    else InputType.PDF if suffix.lower() == ".pdf"
                    else InputType.IMAGE if suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff"]
                    else InputType.CSV
                )

                request = AnalysisRequest(
                    mode=(
                        AnalysisMode.CALCULATION
                        if st.session_state.get("mode") == "Calculation"
                        else AnalysisMode.VERIFICATION
                    ),
                    input_type=input_type_val,
                    source_name=file_path,
                    role_mapping=role_mapping,
                )

                with st.spinner("Running StatVerify AI Engine..."):
                    app_runner = StatVerifyApp()
                    res = app_runner.run(request, selected_test=selected_test)
                    st.session_state["analysis_result"] = res
                    st.session_state["wizard_step"] = 3
                    st.rerun()

    # ----------------------------------------------------------
    # STEP 3: Statistical Results & Verification
    # ----------------------------------------------------------

    elif current_step == 3:
        st.markdown("<div class='panel-header'>📊 Step 3: Statistical Results & Claim Verification</div>", unsafe_allow_html=True)

        result = st.session_state.get("analysis_result")

        if result is None:
            st.warning("No analysis results found. Please execute the pipeline in Step 2.")
            if st.button("⬅️ Back to Step 2"):
                st.session_state["wizard_step"] = 2
                st.rerun()
            st.stop()

        if result.get("error"):
            st.error(f"Analysis Error: {result['error']}")
            if st.button("⬅️ Back to Step 2"):
                st.session_state["wizard_step"] = 2
                st.rerun()
            st.stop()

        # Recommendation & Key Metrics
        if result.get("test_recommendation"):
            rec = result["test_recommendation"]
            selected_test = st.session_state.get("selected_test")
            if selected_test is None:
                st.success(f"**Recommended Test:** {rec.test.value} | **Reason:** {rec.reason}")
            else:
                st.info(f"**Selected Test Override:** {selected_test.value}")

        if result.get("statistical_result"):
            stat_res = result["statistical_result"]

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Test Name", stat_res.test_name or "—")
            with col_m2:
                stat_val = f"{stat_res.statistic:.4f}" if stat_res.statistic is not None else "—"
                st.metric("Statistic", stat_val)
            with col_m3:
                p_val = f"{stat_res.p_value:.4f}" if stat_res.p_value is not None else "—"
                st.metric("p-value", p_val)
            with col_m4:
                es_val = f"{stat_res.effect_size:.4f}" if stat_res.effect_size is not None else "—"
                st.metric("Effect Size", es_val)

            if stat_res.confidence_interval:
                lo, hi = stat_res.confidence_interval
                st.info(f"**95% Confidence Interval:** [{lo:.4f}, {hi:.4f}]")

            # Demographic Multi-Variable Breakdown Table
            if stat_res.additional_results:
                st.markdown("### 📊 Demographic Multi-Variable Breakdown")
                st.write(
                    f"**{stat_res.additional_metrics.get('variables_tested', len(stat_res.additional_results))} "
                    f"demographic variables analysed — "
                    f"{stat_res.additional_metrics.get('significant_variables', 0)} significant (*), "
                    f"{stat_res.additional_metrics.get('non_significant_variables', 0)} non-significant (NS).**"
                )

                table_rows = []
                for entry in stat_res.additional_results:
                    if "error" in entry:
                        table_rows.append({
                            "Variable": entry.get("variable", "?"),
                            "Sub-variable": "-", "Group A No.": "-", "Group A %": "-",
                            "Group B No.": "-", "Group B %": "-", "χ²": "Error", "df": "-", "p-value": "-", "Result": "–",
                        })
                        continue

                    groups = entry.get("groups", ["Experimental", "Control"])
                    group_a = groups[0] if len(groups) > 0 else "Experimental"
                    group_b = groups[1] if len(groups) > 1 else "Control"
                    breakdown = entry.get("breakdown") or [{}]

                    for i, cat in enumerate(breakdown):
                        table_rows.append({
                            "Variable": entry.get("variable", "?") if i == 0 else "",
                            "Sub-variable": cat.get("category", ""),
                            f"{group_a} No.": cat.get(f"{group_a}_n", "-"),
                            f"{group_a} %": cat.get(f"{group_a}_pct", "-"),
                            f"{group_b} No.": cat.get(f"{group_b}_n", "-"),
                            f"{group_b} %": cat.get(f"{group_b}_pct", "-"),
                            "χ²": round(entry["chi_square"], 3) if i == 0 else "",
                            "df": entry["dof"] if i == 0 else "",
                            "p-value": round(entry["p_value"], 3) if i == 0 else "",
                            "Result": ("Sig *" if entry.get("significant") else "NS") if i == 0 else "",
                        })

                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.divider()

        # Assumptions & Verification
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.markdown("### 📋 Statistical Assumptions")
            if result.get("statistical_result") and getattr(result["statistical_result"], "assumptions", None):
                for a in result["statistical_result"].assumptions:
                    badge = "✅ PASS" if a.status.value == "pass" else ("⚠️ WARNING" if a.status.value == "warning" else "❌ FAIL")
                    st.markdown(f"- **{a.name}** (`{badge}`): {a.message}")
            else:
                st.caption("Standard statistical assumptions satisfied.")

        with col_v2:
            st.markdown("### 🔍 Verification & Reported Claims")
            if result.get("verification_result"):
                ver_res = result["verification_result"]
                if ver_res.verified:
                    st.success(f"✅ {ver_res.message}")
                else:
                    st.warning(f"⚠️ {ver_res.message}")

                if ver_res.comparison and ver_res.comparison.items:
                    comp_data = []
                    for item in ver_res.comparison.items:
                        comp_data.append({
                            "Metric": item.metric,
                            "Reported": str(item.reported) if item.reported is not None else "—",
                            "Calculated": str(item.calculated) if item.calculated is not None else "—",
                            "Match": "✅ MATCH" if item.matched else "❌ MISMATCH",
                        })
                    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        st.divider()

        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 2])
        with col_nav1:
            if st.button("⬅️ Back to Test Config", use_container_width=True):
                st.session_state["wizard_step"] = 2
                st.rerun()

        with col_nav3:
            if st.button("Next: View AI Narrative & Download Report 📄 ➡️", use_container_width=True):
                st.session_state["wizard_step"] = 4
                st.rerun()

    # ----------------------------------------------------------
    # STEP 4: AI Explanation & Download PDF Report
    # ----------------------------------------------------------

    elif current_step == 4:
        st.markdown("<div class='panel-header'>📄 Step 4: AI Research Explanation & Official PDF Export</div>", unsafe_allow_html=True)

        result = st.session_state.get("analysis_result")

        if result is None:
            st.warning("No analysis result found. Please execute the pipeline in Step 2.")
            if st.button("⬅️ Back to Step 1"):
                st.session_state["wizard_step"] = 1
                st.rerun()
            st.stop()

        # AI Statistical Interpretation Card
        if result.get("ai_response"):
            st.markdown("### 🤖 AI Research Explanation Narrative")
            ai_resp = result["ai_response"]
            if ai_resp.summary:
                st.info(f"**Executive Summary:** {ai_resp.summary}")
            st.markdown(ai_resp.explanation)

        st.divider()

        # PROMINENT PDF DOWNLOAD HERO CARD
        st.markdown(
            """
            <div class="pdf-download-card">
                <div class="pdf-download-icon">📄</div>
                <div class="pdf-download-title">Official StatVerify AI Verification Report</div>
                <div class="pdf-download-subtitle">
                    Publication-ready PDF document containing complete statistical findings, multi-variable breakdowns, assumption checks, and AI explanations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Retrieve PDF bytes safely from result dict or disk path
        pdf_bytes = result.get("report_pdf")

        if not pdf_bytes and result.get("pdf_path") and os.path.exists(result["pdf_path"]):
            try:
                with open(result["pdf_path"], "rb") as fh:
                    pdf_bytes = fh.read()
            except Exception as ex_pdf:
                st.error(f"Could not read PDF from disk path: {ex_pdf}")

        if pdf_bytes:
            col_pdf1, col_pdf2, col_pdf3 = st.columns([1, 2, 1])
            with col_pdf2:
                st.download_button(
                    label="⬇️ DOWNLOAD OFFICIAL PDF REPORT",
                    data=pdf_bytes,
                    file_name="StatVerify_Official_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="pdf_download_btn_step4",
                )
        else:
            st.warning("PDF report generation pending or unavailable. Check backend logs.")

        st.divider()

        col_reset1, col_reset2, col_reset3 = st.columns([1, 2, 1])
        with col_reset2:
            if st.button("🔄 Start New Statistical Analysis", use_container_width=True):
                st.session_state["wizard_step"] = 1
                st.session_state.pop("analysis_result", None)
                st.session_state.pop("active_file_path", None)
                st.session_state.pop("preview_table", None)
                st.session_state.pop("_role_mapping", None)
                st.rerun()


# ==========================================================
# Vercel Serverless Function Entrypoint Compatibility
# ==========================================================
def app(environ, start_response):
    """WSGI handler for Vercel Serverless deployment detection."""
    status = "200 OK"
    headers = [("Content-Type", "text/html; charset=utf-8")]
    start_response(status, headers)
    html_response = (
        "<!DOCTYPE html>"
        "<html><head><title>StatVerify AI</title></head>"
        "<body style='font-family: sans-serif; padding: 2rem; max-width: 600px; margin: 0 auto; line-height: 1.6;'>"
        "<h2>StatVerify AI Serverless Gateway</h2>"
        "<p>This application is designed as an interactive Streamlit workspace.</p>"
        "<p>For the full interactive UI experience, deploy using Docker on <strong>Render</strong> or <strong>Streamlit Community Cloud</strong>.</p>"
        "</body></html>"
    )
    return [html_response.encode("utf-8")]


application = app
handler = app