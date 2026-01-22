"""
Expert Evaluation Web App for Forensic Analysis Results.

A Streamlit application for experts to review and evaluate LLM reasoning
in forensic image analysis tasks.

Run with: streamlit run webapp/app.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from webapp.utils.data_loader import (
    get_available_results_files,
    load_results,
    filter_results,
)
from webapp.utils.progress_manager import ProgressManager
from webapp.components.sample_viewer import render_sample_viewer
from webapp.components.evaluation_form import render_evaluation_form, render_evaluation_summary
from webapp.components.sidebar import render_sidebar


# Page configuration
st.set_page_config(
    page_title="Expert Evaluation - Forensic Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal layout tweaks; keep Streamlit defaults.
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    
    if "results" not in st.session_state:
        st.session_state.results = []
    
    if "filtered_results" not in st.session_state:
        st.session_state.filtered_results = []
    
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    
    if "progress_manager" not in st.session_state:
        st.session_state.progress_manager = ProgressManager()
    
    if "filters" not in st.session_state:
        st.session_state.filters = {}


def load_file(filename: str):
    """Load a results file and initialize progress."""
    results_dir = Path(__file__).parent.parent / "results"
    filepath = results_dir / filename
    
    st.session_state.results = load_results(filepath)
    st.session_state.filtered_results = st.session_state.results.copy()
    st.session_state.current_file = filename
    st.session_state.current_index = 0
    
    # Load progress for this file
    st.session_state.progress_manager.load_progress(filename)


def apply_filters(filters: dict):
    """Apply filters to the results."""
    results = st.session_state.results
    
    # Extract standard filters
    model = filters.get("model")
    use_tools = filters.get("use_tools")
    prediction = filters.get("prediction")
    label = filters.get("label")
    correct_only = filters.get("correct_only")
    
    # Apply standard filters
    filtered = filter_results(
        results,
        model=model,
        use_tools=use_tools,
        prediction=prediction,
        label=label,
        correct_only=correct_only,
    )
    
    # Apply evaluation status filter
    evaluated = filters.get("evaluated")
    if evaluated is not None:
        evaluated_keys = st.session_state.progress_manager.get_evaluated_keys()
        if evaluated:
            filtered = [r for r in filtered if r.get("_key") in evaluated_keys]
        else:
            filtered = [r for r in filtered if r.get("_key") not in evaluated_keys]
    
    st.session_state.filtered_results = filtered
    st.session_state.filters = filters
    
    # Reset index if out of bounds
    if st.session_state.current_index >= len(filtered):
        st.session_state.current_index = max(0, len(filtered) - 1)


def save_evaluation(evaluation_data: dict):
    """Save an evaluation."""
    pm = st.session_state.progress_manager
    pm.save_evaluation(
        sample_key=evaluation_data["sample_key"],
        reasoning_quality=evaluation_data["reasoning_quality"],
        reasoning_valid=evaluation_data["reasoning_valid"],
        tool_interpretation_correct=evaluation_data["tool_interpretation_correct"],
        fabrication_suspected=evaluation_data["fabrication_suspected"],
        verdict_agree=evaluation_data["verdict_agree"],
        expert_verdict=evaluation_data["expert_verdict"],
        notes=evaluation_data["notes"],
    )


def go_next():
    """Go to next sample."""
    if st.session_state.current_index < len(st.session_state.filtered_results) - 1:
        st.session_state.current_index += 1
        st.rerun()


def go_prev():
    """Go to previous sample."""
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.rerun()


def export_evaluations():
    """Export evaluations to CSV."""
    pm = st.session_state.progress_manager
    if st.session_state.current_file:
        output_path = Path(__file__).parent / "progress" / f"{Path(st.session_state.current_file).stem}_export.csv"
        pm.export_to_csv(output_path)
        st.success(f"Exported to {output_path}")


def main():
    """Main application entry point."""
    initialize_session_state()
    
    # Get available files
    available_files = get_available_results_files()
    
    # Get current state
    results = st.session_state.results
    filtered_results = st.session_state.filtered_results
    current_index = st.session_state.current_index
    pm = st.session_state.progress_manager
    
    # Calculate progress stats
    if results:
        progress_stats = pm.get_progress_stats(len(results))
        evaluated_keys = pm.get_evaluated_keys()
    else:
        progress_stats = {"evaluated": 0, "total": 0, "remaining": 0, "percent_complete": 0}
        evaluated_keys = []
    
    # Render sidebar
    sidebar_result = render_sidebar(
        available_files=available_files,
        current_file=st.session_state.current_file,
        results=results,
        filtered_results=filtered_results,
        current_index=current_index,
        progress_stats=progress_stats,
        evaluated_keys=evaluated_keys,
        on_file_change=load_file,
        on_filter_change=apply_filters,
        on_index_change=lambda idx: setattr(st.session_state, "current_index", idx),
    )
    
    # Handle export action
    if isinstance(sidebar_result, dict) and sidebar_result.get("action") == "export_csv":
        export_evaluations()
    elif isinstance(sidebar_result, dict):
        # Apply filters from sidebar
        apply_filters(sidebar_result)
    
    # Main content area
    if not st.session_state.current_file:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <h1 style="font-size: 3rem; margin-bottom: 20px;">🔬 Expert Evaluation Tool</h1>
            <p style="font-size: 1.2rem; color: #666; margin-bottom: 30px;">
                Review and evaluate LLM reasoning for forensic image analysis
            </p>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; border-radius: 16px; max-width: 600px; margin: 0 auto;">
                <h3 style="color: white; margin-bottom: 16px;">Getting Started</h3>
                <ol style="color: white; text-align: left; opacity: 0.9;">
                    <li>Select a results file from the sidebar</li>
                    <li>Review the LLM's reasoning and tool outputs</li>
                    <li>Provide your expert assessment</li>
                    <li>Progress is automatically saved</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show available files
        if available_files:
            st.markdown("### Available Results Files")
            for f in available_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"📄 **{f.name}**")
                with col2:
                    if st.button("Load", key=f"load_{f.name}", use_container_width=True):
                        load_file(f.name)
                        st.rerun()
        else:
            st.warning("No results files found in the `results/` directory.")
        
        return
    
    # Check if we have results to display
    if not filtered_results:
        st.warning("No samples match the current filters. Try adjusting the filters in the sidebar.")
        
        # Show evaluation summary if available
        if evaluated_keys:
            st.markdown("---")
            st.markdown("### Evaluation Summary")
            render_evaluation_summary(pm.get_all_evaluations())
        return
    
    # Get current sample
    current_result = filtered_results[current_index]
    sample_key = current_result.get("_key", "")
    
    # Two-column layout: Viewer and Evaluation Form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_sample_viewer(
            result=current_result,
            sample_index=current_index,
            total_samples=len(filtered_results),
        )
    
    with col2:
        existing_eval = pm.get_evaluation(sample_key)
        render_evaluation_form(
            sample_key=sample_key,
            existing_evaluation=existing_eval,
            use_tools=current_result.get("use_tools", False),
            on_save=save_evaluation,
            on_next=go_next,
            on_prev=go_prev,
            has_next=current_index < len(filtered_results) - 1,
            has_prev=current_index > 0,
        )
    
    # Footer with summary
    st.markdown("---")
    st.markdown("### Session Summary")
    render_evaluation_summary(pm.get_all_evaluations())


if __name__ == "__main__":
    main()
