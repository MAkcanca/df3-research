"""
Sidebar component for navigation and filtering.

Provides file selection, filters, and progress display.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st


def render_sidebar(
    available_files: List[Path],
    current_file: Optional[str],
    results: List[Dict[str, Any]],
    filtered_results: List[Dict[str, Any]],
    current_index: int,
    progress_stats: Dict[str, Any],
    evaluated_keys: List[str],
    on_file_change: Callable[[str], None],
    on_filter_change: Callable[[Dict[str, Any]], None],
    on_index_change: Callable[[int], None],
) -> Dict[str, Any]:
    """
    Render the sidebar with navigation and filters.
    
    Args:
        available_files: List of available JSONL files.
        current_file: Currently selected file.
        results: All loaded results.
        filtered_results: Currently filtered results.
        current_index: Current sample index.
        progress_stats: Progress statistics.
        evaluated_keys: List of evaluated sample keys.
        on_file_change: Callback when file selection changes.
        on_filter_change: Callback when filters change.
        on_index_change: Callback when index changes.
    
    Returns:
        Current filter settings.
    """
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 16px 0;">
        <h1 style="margin: 0; font-size: 1.5rem;">🔬 Expert Evaluation</h1>
        <p style="margin: 4px 0 0 0; opacity: 0.7; font-size: 0.9rem;">Forensic Analysis Review</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # File Selection
    st.sidebar.markdown("### 📁 Results File")
    
    file_options = ["Select a file..."] + [f.name for f in available_files]
    current_selection = current_file if current_file in file_options else "Select a file..."
    
    selected_file = st.sidebar.selectbox(
        "Select results file",
        options=file_options,
        index=file_options.index(current_selection) if current_selection in file_options else 0,
        key="file_selector",
        label_visibility="collapsed",
    )
    
    if selected_file != "Select a file..." and selected_file != current_file:
        on_file_change(selected_file)
        st.rerun()
    
    # Progress Display
    if results:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Progress")
        
        evaluated = progress_stats.get("evaluated", 0)
        total = progress_stats.get("total", 0)
        percent = progress_stats.get("percent_complete", 0)
        
        st.sidebar.progress(percent / 100)
        st.sidebar.markdown(f"**{evaluated}** of **{total}** evaluated ({percent:.1f}%)")
        
        remaining = progress_stats.get("remaining", 0)
        if remaining > 0:
            st.sidebar.caption(f"{remaining} samples remaining")
        else:
            st.sidebar.success("All samples evaluated! 🎉")
    
    # Filters
    filters = {}
    if results:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔍 Filters")
        
        # Get unique values for filters
        models = sorted(set(r.get("model", "") for r in results))
        predictions = sorted(set(r.get("prediction", "") for r in results))
        labels = sorted(set(r.get("label", "") for r in results))
        
        # Model filter
        model_options = ["All Models"] + models
        selected_model = st.sidebar.selectbox(
            "Model",
            options=model_options,
            key="filter_model",
        )
        if selected_model != "All Models":
            filters["model"] = selected_model
        
        # Tools filter
        tools_options = ["Both", "With Tools", "Without Tools"]
        selected_tools = st.sidebar.selectbox(
            "Tools",
            options=tools_options,
            key="filter_tools",
        )
        if selected_tools == "With Tools":
            filters["use_tools"] = True
        elif selected_tools == "Without Tools":
            filters["use_tools"] = False
        
        # Prediction filter
        pred_options = ["All"] + predictions
        selected_pred = st.sidebar.selectbox(
            "Prediction",
            options=pred_options,
            key="filter_prediction",
        )
        if selected_pred != "All":
            filters["prediction"] = selected_pred
        
        # Label filter
        label_options = ["All"] + labels
        selected_label = st.sidebar.selectbox(
            "Ground Truth",
            options=label_options,
            key="filter_label",
        )
        if selected_label != "All":
            filters["label"] = selected_label
        
        # Correctness filter
        correct_options = ["All", "Correct Only", "Incorrect Only"]
        selected_correct = st.sidebar.selectbox(
            "Result",
            options=correct_options,
            key="filter_correct",
        )
        if selected_correct == "Correct Only":
            filters["correct_only"] = True
        elif selected_correct == "Incorrect Only":
            filters["correct_only"] = False
        
        # Evaluation status filter
        eval_options = ["All", "Not Evaluated", "Already Evaluated"]
        selected_eval = st.sidebar.selectbox(
            "Evaluation Status",
            options=eval_options,
            key="filter_eval_status",
        )
        if selected_eval == "Not Evaluated":
            filters["evaluated"] = False
        elif selected_eval == "Already Evaluated":
            filters["evaluated"] = True
    
    # Navigation
    if filtered_results:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧭 Navigation")
        
        # Sample number input
        nav_key = "nav_sample_num"
        if st.session_state.get(nav_key) != current_index + 1:
            st.session_state[nav_key] = current_index + 1
        sample_num = st.sidebar.number_input(
            "Go to sample #",
            min_value=1,
            max_value=len(filtered_results),
            value=current_index + 1,
            key=nav_key,
        )
        
        if sample_num - 1 != current_index:
            on_index_change(sample_num - 1)
            st.rerun()
        
        # Quick navigation buttons
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("⏮ First", use_container_width=True, key="nav_first"):
                on_index_change(0)
                st.rerun()
        
        with col2:
            if st.button("Last ⏭", use_container_width=True, key="nav_last"):
                on_index_change(len(filtered_results) - 1)
                st.rerun()
        
        # Jump to next unevaluated
        if evaluated_keys:
            filtered_keys = [r.get("_key", "") for r in filtered_results]
            unevaluated_indices = [
                i for i, key in enumerate(filtered_keys)
                if key not in evaluated_keys
            ]
            
            if unevaluated_indices:
                # Find next unevaluated after current index
                next_uneval = None
                for idx in unevaluated_indices:
                    if idx > current_index:
                        next_uneval = idx
                        break
                
                if next_uneval is None and unevaluated_indices:
                    next_uneval = unevaluated_indices[0]
                
                if next_uneval is not None:
                    if st.sidebar.button("⏩ Next Unevaluated", use_container_width=True, key="nav_next_uneval"):
                        on_index_change(next_uneval)
                        st.rerun()
    
    # Export
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Export")
    
    if st.sidebar.button("Export to CSV", use_container_width=True, key="export_csv"):
        return {"action": "export_csv", **filters}
    
    return filters


def render_file_info(filepath: Path, results_count: int) -> None:
    """Render information about the loaded file."""
    st.sidebar.markdown(f"**📄 {filepath.name}**")
    st.sidebar.caption(f"{results_count} samples loaded")
