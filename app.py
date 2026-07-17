"""
StatVerify AI

Main Streamlit Application
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st

from src.models import (
    AnalysisMode,
    AnalysisRequest,
    InputType,
    StatisticalTest,
)

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
            "ai_response": None,
            "error": None,
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
            "error": result.get("error"),
        }


# ==========================================================
# Streamlit UI
# ==========================================================

if __name__ == "__main__" or hasattr(st, "_is_running_with_streamlit"):

    # ----------------------------------------------------------
    # Page Configuration
    # ----------------------------------------------------------

    st.set_page_config(
        page_title="StatVerify AI",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 StatVerify AI")
    st.caption(
        "Automatic Statistical Analysis, Verification and AI Explanation"
    )

    st.divider()

    # ----------------------------------------------------------
    # Sidebar
    # ----------------------------------------------------------

    with st.sidebar:

        st.header("Analysis Settings")

        mode = st.radio(
            "Mode",
            [
                "Calculation",
                "Verification",
            ],
        )

        uploaded_file = st.file_uploader(
            "Upload CSV / Excel",
            type=["csv", "xlsx"],
        )

        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as tmp:

                tmp.write(uploaded_file.read())
                file_path = tmp.name

        else:
            file_path = None

        st.subheader("Statistical Test")

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
            "Choose Statistical Test",
            list(test_options.keys()),
        )

        selected_test = test_options[selected_test_name]

        run_button = st.button(
            "Run Analysis",
            use_container_width=True,
        )

    # ----------------------------------------------------------
    # Main Area
    # ----------------------------------------------------------

    if not run_button:

        st.info(
            "Upload a dataset and click **Run Analysis**."
        )

        st.stop()

    if file_path is not None:

        suffix = os.path.splitext(uploaded_file.name)[1]

        request = AnalysisRequest(
            mode=(
                AnalysisMode.CALCULATION
                if mode == "Calculation"
                else AnalysisMode.VERIFICATION
            ),
            input_type=(
                InputType.CSV
                if suffix.lower() == ".csv"
                else InputType.EXCEL
            ),
            source_name=file_path,
        )

        with st.spinner("Running workflow..."):

            app = StatVerifyApp()
            result = app.run(request, selected_test=selected_test)

        if result.get("error"):
            st.error(result["error"])
            st.stop()

        st.success("Analysis complete!")

        if result.get("test_recommendation"):

            recommendation = result["test_recommendation"]

            if selected_test is None:
                st.success(
                    f"Recommended Test: **{recommendation.test.value}**"
                )
                st.caption(recommendation.reason)
            else:
                st.info(
                    f"Selected Test: **{selected_test.value}**"
                )

        if result.get("statistical_result"):
            st.subheader("Statistical Result")
            stat_res = result["statistical_result"]
            if stat_res.additional_metrics.get("demographics"):
                st.write("### Demographic Analysis Summary")
                for var, metrics in stat_res.additional_metrics["demographics"].items():
                    st.info(f"**{var}**")
                    st.write(f"χ² = {metrics['chi2']:.2f}, p = {metrics['p']:.3f} (df={int(metrics['dof'])})")
            else:
                st.json(stat_res.model_dump())

        if result.get("ai_response"):
            st.subheader("AI Explanation")
            st.write(result["ai_response"].explanation)

        # Generate PDF Report on-demand
        from src.reports import report_engine

        st.subheader("Report Export")
        if st.button("Generate PDF Report"):
            pdf_path = report_engine.generate(result)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download PDF",
                    f,
                    "StatVerify_Report.pdf",
                    mime="application/pdf",
                )

    else:

        st.info("Please upload a dataset.")