"""
Sample viewer component for displaying forensic analysis results.

Shows the image, LLM reasoning, tool outputs, and verdict information.
"""

import base64
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st


def load_image_as_base64(image_path: str) -> Optional[str]:
    """Load an image file and return as base64 string."""
    try:
        path = Path(image_path)
        if not path.exists():
            return None
        
        with open(path, "rb") as f:
            data = f.read()
        
        # Determine MIME type
        suffix = path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/jpeg")
        
        return f"data:{mime_type};base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        st.error(f"Could not load image: {e}")
        return None


def render_verdict_badge(prediction: str, label: str, confidence: float) -> None:
    """Render verdict badge with correct/incorrect indicator."""
    is_correct = prediction.lower() == label.lower()
    
    # Color coding
    if is_correct:
        verdict_color = "green"
        icon = "✓"
    else:
        verdict_color = "red"
        icon = "✗"
    
    # Prediction color
    pred_colors = {
        "real": "#28a745",
        "fake": "#dc3545",
        "uncertain": "#ffc107",
    }
    pred_color = pred_colors.get(prediction.lower(), "#6c757d")
    
    # Hover-to-reveal styling: both Ground Truth and Correctness reveal together
    # Correctness color also changes on hover
    correctness_text = f"{icon} Correct" if is_correct else f"{icon} Incorrect"
    st.markdown(
        f"""
        <style>
        .df3-spoiler-group {{ display: flex; gap: 12px; cursor: pointer; }}
        .df3-spoiler-group .df3-secret {{ display: none; }}
        .df3-spoiler-group .df3-placeholder {{ display: inline; }}
        .df3-spoiler-group:hover .df3-secret {{ display: inline; }}
        .df3-spoiler-group:hover .df3-placeholder {{ display: none; }}
        .df3-spoiler-group .df3-correctness {{ background: #e9ecef; color: #333; }}
        .df3-spoiler-group:hover .df3-correctness {{ background: {verdict_color}; color: white; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 16px;">
        <div style="padding: 8px 16px; border-radius: 8px; background: {pred_color}; color: white; font-weight: bold;">
            Prediction: {prediction.upper()}
        </div>
        <div style="padding: 8px 16px; border-radius: 8px; background: #17a2b8; color: white;">
            Confidence: {confidence:.0%}
        </div>
        <div class="df3-spoiler-group">
            <div style="padding: 8px 16px; border-radius: 8px; background: #e9ecef; color: #333;">
                Ground Truth:
                <span class="df3-placeholder">reveal</span>
                <span class="df3-secret">{label.upper()}</span>
            </div>
            <div class="df3-correctness" style="padding: 8px 16px; border-radius: 8px;">
                Correctness:
                <span class="df3-placeholder">reveal</span>
                <span class="df3-secret">{correctness_text}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_tool_results(result: Dict[str, Any]) -> None:
    """Render tool usage and results in collapsible sections."""
    tool_usage = result.get("tool_usage", [])
    tool_results = result.get("tool_results", [])
    tool_details = result.get("tool_details", [])
    
    if not tool_usage:
        st.info("No forensic tools were used for this analysis.")
        return
    
    st.markdown(f"**Tools Used:** {', '.join(tool_usage)}")
    
    # Show timing summary
    total_time = sum(d.get("seconds", 0) for d in tool_details)
    st.markdown(f"**Total Tool Time:** {total_time:.2f}s")
    
    # Show each tool result
    for tr in tool_results:
        tool_name = tr.get("tool", "unknown")
        status = tr.get("status", "unknown")
        seconds = tr.get("seconds", 0)
        parsed = tr.get("parsed", {})
        
        status_icon = "✓" if status == "completed" else "✗"
        
        with st.expander(f"{status_icon} {tool_name} ({seconds:.3f}s)", expanded=False):
            if tr.get("error"):
                st.error(f"Error: {tr['error']}")
            
            # Display key metrics based on tool type
            if tool_name == "perform_trufor":
                prob = parsed.get("manipulation_probability", "N/A")
                if isinstance(prob, (int, float)):
                    color = "red" if prob > 0.5 else "green"
                    st.markdown(f"**Manipulation Probability:** <span style='color: {color}; font-weight: bold;'>{prob:.3f}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Manipulation Probability:** {prob}")
            
            elif tool_name == "perform_ela":
                score = parsed.get("ela_anomaly_score", "N/A")
                if isinstance(score, (int, float)):
                    color = "red" if score > 2.0 else "green"
                    st.markdown(f"**ELA Anomaly Score:** <span style='color: {color}; font-weight: bold;'>{score:.3f}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**ELA Anomaly Score:** {score}")
                st.markdown(f"**ELA Mean:** {parsed.get('ela_mean', 'N/A')}")
            
            elif tool_name == "metadata":
                exif = parsed.get("exif", {})
                c2pa = parsed.get("c2pa", {})
                st.markdown(f"**EXIF Present:** {exif.get('present', False)}")
                st.markdown(f"**C2PA Present:** {c2pa.get('present', False)}")
                if exif.get("summary"):
                    st.json(exif["summary"])
            
            # Show full parsed output
            st.markdown("**Full Output:**")
            st.json(parsed)


def render_sample_viewer(result: Dict[str, Any], sample_index: int, total_samples: int) -> None:
    """
    Render the complete sample viewer.
    
    Args:
        result: The result dictionary for this sample.
        sample_index: Current sample index (0-based).
        total_samples: Total number of samples.
    """
    sample_id = result.get("id", "unknown")
    model = result.get("model", "unknown")
    use_tools = result.get("use_tools", False)
    trial = result.get("trial", 0)
    
    # Header
    st.subheader(f"Sample: {sample_id}")
    st.caption(
        f"Model: {model} | Tools: {'Yes' if use_tools else 'No'} | Trial: {trial} | "
        f"Sample {sample_index + 1} of {total_samples}"
    )
    
    # Verdict badges
    render_verdict_badge(
        result.get("prediction", "unknown"),
        result.get("label", "unknown"),
        result.get("confidence", 0.0)
    )
    
    # Two column layout: Image and Summary
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Image")
        image_path = result.get("image", "")
        if image_path:
            image_data = load_image_as_base64(image_path)
            if image_data:
                st.markdown(f"""
                <div style="border: 2px solid #ddd; border-radius: 8px; padding: 8px; background: #f8f9fa;">
                    <img src="{image_data}" style="max-width: 100%; height: auto; border-radius: 4px;">
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"Image not found: {image_path}")
        else:
            st.warning("No image path provided")
    
    with col2:
        st.markdown("### Summary")
        
        # Rationale
        rationale = result.get("rationale", "No rationale provided")
        st.markdown("**Rationale**")
        st.write(rationale)
        
        # Visual description
        visual_desc = result.get("visual_description", "No description provided")
        st.markdown("**Visual Description**")
        st.write(visual_desc)
        
        # Forensic summary
        forensic_summary = result.get("forensic_summary", "No tools used")
        st.markdown("**Forensic Summary**")
        st.write(forensic_summary)
    
    # Full reasoning section
    st.markdown("---")
    st.markdown("### Full LLM Reasoning")
    
    raw_analysis = result.get("raw_analysis", "No analysis available")
    with st.expander("Click to expand full reasoning", expanded=True):
        st.markdown(raw_analysis)
    
    # Tool results section
    st.markdown("---")
    st.markdown("### Forensic Tool Results")
    render_tool_results(result)
    
    # Prompts section (collapsible for transparency)
    st.markdown("---")
    prompts = result.get("prompts", {})
    if prompts:
        with st.expander("View Prompts Used (for transparency)", expanded=False):
            for prompt_name, prompt_text in prompts.items():
                st.markdown(f"**{prompt_name}:**")
                st.code(prompt_text[:2000] + "..." if len(prompt_text) > 2000 else prompt_text)
    
    # Timing information
    timings = result.get("timings", {})
    latency = result.get("latency_seconds", 0)
    if timings or latency:
        with st.expander("Timing Information", expanded=False):
            if latency:
                st.markdown(f"**Total Latency:** {latency:.2f}s")
            for timing_name, timing_value in timings.items():
                st.markdown(f"**{timing_name}:** {timing_value:.2f}s")
