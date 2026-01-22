"""
Evaluation form component for expert assessments.

Provides input fields for experts to evaluate LLM reasoning quality.
"""

from typing import Any, Callable, Dict, Optional

import streamlit as st


def render_evaluation_form(
    sample_key: str,
    existing_evaluation: Optional[Dict[str, Any]],
    use_tools: bool,
    on_save: Callable[[Dict[str, Any]], None],
    on_next: Callable[[], None],
    on_prev: Callable[[], None],
    has_next: bool,
    has_prev: bool,
) -> None:
    """
    Render the evaluation form for expert input.
    
    Args:
        sample_key: Unique key for the current sample.
        existing_evaluation: Existing evaluation data if already evaluated.
        use_tools: Whether tools were used for this sample.
        on_save: Callback when saving evaluation.
        on_next: Callback to go to next sample.
        on_prev: Callback to go to previous sample.
        has_next: Whether there is a next sample.
        has_prev: Whether there is a previous sample.
    """
    st.subheader("Expert Evaluation")
    st.caption("Assess the quality and validity of the LLM's reasoning")
    
    # Status indicator
    if existing_evaluation:
        st.success("✓ This sample has been evaluated")
        timestamp = existing_evaluation.get("timestamp", "Unknown")
        st.caption(f"Last evaluated: {timestamp}")
    else:
        st.warning("⚠ This sample has not been evaluated yet")
    
    # Form key for this specific sample
    form_key = f"eval_form_{sample_key}"
    
    # Pre-fill with existing values or defaults
    defaults = {
        "reasoning_quality": existing_evaluation.get("reasoning_quality", 3) if existing_evaluation else 3,
        "reasoning_valid": existing_evaluation.get("reasoning_valid", True) if existing_evaluation else True,
        "tool_interpretation_correct": existing_evaluation.get("tool_interpretation_correct", True) if existing_evaluation else True,
        "fabrication_suspected": existing_evaluation.get("fabrication_suspected", False) if existing_evaluation else False,
        "verdict_agree": existing_evaluation.get("verdict_agree", True) if existing_evaluation else True,
        "expert_verdict": existing_evaluation.get("expert_verdict", "uncertain") if existing_evaluation else "uncertain",
        "notes": existing_evaluation.get("notes", "") if existing_evaluation else "",
    }
    
    # Reasoning Quality Rating
    st.markdown("#### 1. Reasoning Quality")
    st.caption("Rate the overall quality of the LLM's reasoning (1=Poor, 5=Excellent)")
    
    reasoning_quality = st.slider(
        "Quality Score",
        min_value=1,
        max_value=5,
        value=defaults["reasoning_quality"],
        key=f"quality_{sample_key}",
        help="1=Poor reasoning, 2=Below average, 3=Average, 4=Good, 5=Excellent",
    )
    
    # Display quality description
    quality_descriptions = {
        1: "Poor - Reasoning is flawed, irrelevant, or nonsensical",
        2: "Below Average - Significant issues with logic or relevance",
        3: "Average - Acceptable reasoning with some weaknesses",
        4: "Good - Sound reasoning with minor issues",
        5: "Excellent - Clear, logical, and well-supported reasoning",
    }
    st.info(quality_descriptions[reasoning_quality])
    
    st.markdown("---")
    
    # Validity Assessments
    st.markdown("#### 2. Validity Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        reasoning_valid = st.checkbox(
            "Reasoning is scientifically sound",
            value=defaults["reasoning_valid"],
            key=f"valid_{sample_key}",
            help="Check if the reasoning follows sound scientific/forensic principles",
        )
        
        fabrication_suspected = st.checkbox(
            "Fabrication suspected",
            value=defaults["fabrication_suspected"],
            key=f"fabrication_{sample_key}",
            help="Check if the LLM appears to have made up facts or hallucinated",
        )
    
    with col2:
        # Only show tool interpretation if tools were used
        if use_tools:
            tool_interpretation_correct = st.checkbox(
                "Tool results correctly interpreted",
                value=defaults["tool_interpretation_correct"],
                key=f"tool_correct_{sample_key}",
                help="Check if the LLM correctly interpreted the forensic tool outputs",
            )
        else:
            tool_interpretation_correct = None
            st.caption("(No tools used for this sample)")
        
        verdict_agree = st.checkbox(
            "Agree with verdict",
            value=defaults["verdict_agree"],
            key=f"agree_{sample_key}",
            help="Check if you agree with the LLM's final verdict",
        )
    
    st.markdown("---")
    
    # Expert's Own Verdict
    st.markdown("#### 3. Expert Verdict")
    st.caption("What is your own assessment of this image?")
    
    verdict_options = ["real", "fake", "uncertain"]
    expert_verdict = st.radio(
        "Your verdict",
        options=verdict_options,
        index=verdict_options.index(defaults["expert_verdict"]),
        key=f"expert_verdict_{sample_key}",
        horizontal=True,
    )
    
    st.markdown("---")
    
    # Notes
    st.markdown("#### 4. Notes")
    notes = st.text_area(
        "Additional observations or comments",
        value=defaults["notes"],
        key=f"notes_{sample_key}",
        height=120,
        placeholder="Enter any additional observations, issues noticed, or comments about the reasoning...",
    )
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Previous", disabled=not has_prev, use_container_width=True, key=f"prev_{sample_key}"):
            on_prev()
    
    with col2:
        if st.button("💾 Save Evaluation", type="primary", use_container_width=True, key=f"save_{sample_key}"):
            evaluation_data = {
                "sample_key": sample_key,
                "reasoning_quality": reasoning_quality,
                "reasoning_valid": reasoning_valid,
                "tool_interpretation_correct": tool_interpretation_correct,
                "fabrication_suspected": fabrication_suspected,
                "verdict_agree": verdict_agree,
                "expert_verdict": expert_verdict,
                "notes": notes,
            }
            on_save(evaluation_data)
            st.success("✓ Evaluation saved!")
            st.rerun()
    
    with col3:
        if st.button("Next →", disabled=not has_next, use_container_width=True, key=f"next_{sample_key}"):
            on_next()
    
    # Save & Navigate buttons
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save & Previous", disabled=not has_prev, use_container_width=True, key=f"save_prev_{sample_key}"):
            evaluation_data = {
                "sample_key": sample_key,
                "reasoning_quality": reasoning_quality,
                "reasoning_valid": reasoning_valid,
                "tool_interpretation_correct": tool_interpretation_correct,
                "fabrication_suspected": fabrication_suspected,
                "verdict_agree": verdict_agree,
                "expert_verdict": expert_verdict,
                "notes": notes,
            }
            on_save(evaluation_data)
            on_prev()
    
    with col2:
        if st.button("💾 Save & Next", disabled=not has_next, use_container_width=True, key=f"save_next_{sample_key}"):
            evaluation_data = {
                "sample_key": sample_key,
                "reasoning_quality": reasoning_quality,
                "reasoning_valid": reasoning_valid,
                "tool_interpretation_correct": tool_interpretation_correct,
                "fabrication_suspected": fabrication_suspected,
                "verdict_agree": verdict_agree,
                "expert_verdict": expert_verdict,
                "notes": notes,
            }
            on_save(evaluation_data)
            on_next()


def render_evaluation_summary(evaluations: Dict[str, Dict[str, Any]]) -> None:
    """
    Render a summary of all evaluations.
    
    Args:
        evaluations: Dictionary of all evaluations.
    """
    if not evaluations:
        st.info("No evaluations yet.")
        return
    
    total = len(evaluations)
    
    # Calculate statistics
    avg_quality = sum(e.get("reasoning_quality", 0) for e in evaluations.values()) / total
    valid_count = sum(1 for e in evaluations.values() if e.get("reasoning_valid", False))
    fabrication_count = sum(1 for e in evaluations.values() if e.get("fabrication_suspected", False))
    agree_count = sum(1 for e in evaluations.values() if e.get("verdict_agree", False))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Quality", f"{avg_quality:.1f}/5")
    
    with col2:
        st.metric("Valid Reasoning", f"{valid_count}/{total}")
    
    with col3:
        st.metric("Fabrications", f"{fabrication_count}/{total}")
    
    with col4:
        st.metric("Agree with Verdict", f"{agree_count}/{total}")
