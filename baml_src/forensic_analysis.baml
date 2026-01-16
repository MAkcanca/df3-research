// Forensic Analysis BAML Functions
// Following the multi-step approach to avoid reasoning degradation:
// Step 1: Reasoning (unstructured) - allows LLM to reason freely
// Step 2: Structuring (structured) - extracts structured data from reasoning

// Enum for verdict
enum Verdict {
  REAL @description("Image appears authentic and natural")
  FAKE @description("Image appears AI-generated, synthetic, or manipulated")
  UNCERTAIN @description("Insufficient evidence or conflicting indicators")
}

// Enum for confidence level (avoiding numeric intervals per BAML best practices)
enum ConfidenceLevel {
  HIGH @description("High confidence in the verdict")
  MEDIUM @description("Medium confidence in the verdict")
  LOW @description("Low confidence in the verdict")
}

// Structured output model (used only in structuring step, not during reasoning)
class ForensicAnalysisResult {
  verdict Verdict
  confidence float @assert(between_0_and_1, {{ this >= 0.0 and this <= 1.0 }})
  rationale string @description("Brief justification for the verdict (max 80 words)")
  visual_description string @description("Description of what is in the image")
  forensic_summary string @description("Summary of forensic tools used or 'No tools used'")
  full_text string @description("Complete formatted narrative combining all sections")
}

// Step 1: Vision-only reasoning (UNSTRUCTURED - allows free reasoning)
// This function returns unstructured text to avoid constraining reasoning
function AnalyzeImageVisionOnly(image: image) -> string {
  client DynamicForensicClient
  prompt #"
    You are a forensic image analyst specializing in detecting AI-generated or manipulated images.
    
    CRITICAL: You MUST always start your analysis by describing what is actually in the image - the subjects, scene, objects, people, animals, environment, etc. Do NOT skip directly to forensic metrics.
    
    Analyze this image and decide one of three outcomes:
    - real: authentic photograph
    - fake: AI-generated, synthetic, or manipulated
    - uncertain: inconclusive → route to human/manual review
    
    Safety rule: Do NOT output "real" unless you have affirmative reasons it is authentic. Prefer "uncertain" over guessing.
    
    Provide your analysis in MARKDOWN format with this structure:
    ### Visual Description
    - Describe what is visibly in the image (subjects, scene, objects, people/animals, environment, colors, composition)
    - Analyze lighting: sources, direction, intensity, shadows, reflections, consistency
    - Check physics: perspective, shadows, reflections, physical interactions, textures
    - Note any visual anomalies or inconsistencies you observe
    
    ### Forensic Analysis
    - Vision-only pass; note "No tools used" and list any visual cues for/against synthesis
    
    ### Conclusion
    - State if the image looks synthetic/AI vs natural, and why (refer to observations above)
    - Include a line "Verdict: real | fake | uncertain" (uncertain means inconclusive/manual review)
    
    ### Confidence
    - State High / Medium / Low with a brief justification
    - Include "Confidence (0-1): <value between 0 and 1>"
    
    Think through your reasoning step by step. Do not constrain your thinking - provide detailed analysis.
    
    {{ _.role("user") }} {{ image }}
  "#
}

// Note: Agent reasoning with tools is handled by LangChain/LangGraph in the Python code
// BAML is used for vision-only analysis and structuring steps to avoid reasoning degradation

// Step 3: Structuring function (STRUCTURED - extracts structured data from unstructured reasoning)
// This function takes the unstructured reasoning output and extracts structured data
// This separation ensures reasoning quality is not degraded by format constraints
function StructureForensicAnalysis(reasoning_output: string) -> ForensicAnalysisResult {
  client DynamicForensicClient
  prompt #"
    Extract structured information from this forensic analysis reasoning output.
    
    The analysis may contain:
    - A visual description of the image
    - Forensic tool results or summaries
    - A conclusion with a verdict (real/fake/uncertain)
    - A confidence level and value
    
    Extract the following information:
    - verdict: The final verdict (real, fake, or uncertain). Treat "inconclusive" / "cannot determine" as "uncertain".
    - confidence: A float between 0 and 1
    - rationale: A brief justification (max 80 words)
    - visual_description: Description of what's in the image
    - forensic_summary: Summary of tools used or "No tools used"
    - full_text: The complete formatted narrative from the reasoning output
    
    Reasoning output:
    {{ reasoning_output }}
    
    {{ ctx.output_format }}
  "#
}

// Helper function for vision-only analysis that returns structured output
// NOTE: This function directly returns structured output, which may cause reasoning degradation.
// For best results, prefer the two-step approach: AnalyzeImageVisionOnly + StructureForensicAnalysis
// This function is provided for convenience in simple cases where reasoning complexity is low.
function AnalyzeImageVisionOnlyStructured(image: image) -> ForensicAnalysisResult {
  client DynamicForensicClient
  prompt #"
    You are a forensic image analyst. Analyze this image and assess whether it appears AI-generated, synthetic, or a deepfake.
    
    CRITICAL: You MUST always start your analysis by describing what is actually in the image - the subjects, scene, objects, people, animals, environment, etc.
    
    Think through your reasoning step by step. Consider:
    1. Visual description: What is in the image, lighting, physics
    2. Synthesis indicators: Visual cues for/against synthesis
    3. Verdict: real, fake, or uncertain (uncertain means inconclusive/manual review)
    4. Confidence: A value between 0 and 1
    5. Rationale: Brief justification (max 80 words)
    
    Safety rule: Do NOT output "real" unless you have affirmative reasons it is authentic. Prefer "uncertain" over guessing.
    
    After reasoning through these points, provide your structured answer.
    
    {{ _.role("user") }} {{ image }}
    
    {{ ctx.output_format }}
  "#
}
