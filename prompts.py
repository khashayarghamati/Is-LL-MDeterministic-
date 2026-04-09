"""
20 Prompts on "AI in Healthcare" — same topic, varied angles.

Every prompt shares a single OUTPUT_TEMPLATE so responses are structurally
comparable across models and repetitions.
"""

TOPIC = "Artificial Intelligence in Healthcare"

OUTPUT_TEMPLATE = """
You must respond ONLY with a valid JSON object. Do not include any text,
explanation, or markdown outside the JSON.  Use exactly this structure:

{
    "summary": "<2-3 sentence overview>",
    "key_points": [
        "<point 1>",
        "<point 2>",
        "<point 3>",
        "<point 4>",
        "<point 5>"
    ],
    "challenges": [
        "<challenge 1>",
        "<challenge 2>",
        "<challenge 3>"
    ],
    "potential_impact": "<1-2 sentences about future impact>",
    "confidence_level": "<high OR medium OR low>",
    "estimated_timeline": "<short-term OR medium-term OR long-term>"
}
""".strip()

# ---------------------------------------------------------------------------
# 20 prompts — all about AI in Healthcare, each from a different angle
# ---------------------------------------------------------------------------
PROMPTS: list[dict[str, str]] = [
    {
        "id": "P01",
        "angle": "General overview",
        "question": (
            "Explain the current state of artificial intelligence applications "
            "in healthcare and their overall impact on patient outcomes."
        ),
    },
    {
        "id": "P02",
        "angle": "Medical diagnostics",
        "question": (
            "Analyze how AI-powered diagnostic tools are changing the accuracy "
            "and speed of disease detection in clinical settings."
        ),
    },
    {
        "id": "P03",
        "angle": "Drug discovery",
        "question": (
            "Discuss how artificial intelligence is accelerating pharmaceutical "
            "drug discovery and reducing time-to-market for new treatments."
        ),
    },
    {
        "id": "P04",
        "angle": "Patient data privacy",
        "question": (
            "Evaluate the privacy and security challenges that arise from "
            "using AI systems to process sensitive patient health data."
        ),
    },
    {
        "id": "P05",
        "angle": "Rural healthcare access",
        "question": (
            "Describe how AI technologies can improve healthcare accessibility "
            "in rural and underserved communities."
        ),
    },
    {
        "id": "P06",
        "angle": "Mental health",
        "question": (
            "Assess the potential of AI-driven tools for mental health "
            "screening, therapy assistance, and psychological well-being monitoring."
        ),
    },
    {
        "id": "P07",
        "angle": "Cost reduction",
        "question": (
            "Analyze how artificial intelligence can reduce operational costs "
            "in healthcare systems while maintaining or improving quality of care."
        ),
    },
    {
        "id": "P08",
        "angle": "Robotic surgery",
        "question": (
            "Evaluate the current capabilities and future potential of "
            "AI-assisted robotic surgery systems in modern operating rooms."
        ),
    },
    {
        "id": "P09",
        "angle": "Personalized medicine",
        "question": (
            "Discuss how AI enables personalized medicine by tailoring "
            "treatments to individual patient genetic profiles and health histories."
        ),
    },
    {
        "id": "P10",
        "angle": "Electronic health records",
        "question": (
            "Explain how AI can improve the management and analysis of "
            "electronic health records to enhance clinical decision-making."
        ),
    },
    {
        "id": "P11",
        "angle": "Clinical trials",
        "question": (
            "Analyze the impact of AI on clinical trial design, patient "
            "recruitment, and outcome prediction in pharmaceutical research."
        ),
    },
    {
        "id": "P12",
        "angle": "Ethics",
        "question": (
            "Discuss the key ethical dilemmas surrounding the deployment "
            "of AI systems in healthcare decision-making and patient care."
        ),
    },
    {
        "id": "P13",
        "angle": "Medical imaging",
        "question": (
            "Evaluate how deep learning models are advancing medical image "
            "analysis in radiology, pathology, and dermatology."
        ),
    },
    {
        "id": "P14",
        "angle": "Elderly care",
        "question": (
            "Describe the applications of AI in elderly care, including "
            "fall detection, medication management, and cognitive health monitoring."
        ),
    },
    {
        "id": "P15",
        "angle": "Pandemic response",
        "question": (
            "Assess how AI systems contributed to pandemic preparedness, "
            "epidemiological modeling, and vaccine development during recent "
            "global health crises."
        ),
    },
    {
        "id": "P16",
        "angle": "Medical education",
        "question": (
            "Discuss how AI-powered simulations and virtual patients are "
            "transforming medical education and surgical training programs."
        ),
    },
    {
        "id": "P17",
        "angle": "Billing and insurance",
        "question": (
            "Analyze the potential of AI to streamline healthcare billing, "
            "insurance claims processing, and fraud detection."
        ),
    },
    {
        "id": "P18",
        "angle": "Rare diseases",
        "question": (
            "Evaluate how AI algorithms can improve the detection and "
            "diagnosis of rare diseases that are often missed by traditional "
            "diagnostic methods."
        ),
    },
    {
        "id": "P19",
        "angle": "Preventive care",
        "question": (
            "Discuss how AI can enhance preventive healthcare through early "
            "risk prediction, lifestyle recommendations, and continuous "
            "health monitoring."
        ),
    },
    {
        "id": "P20",
        "angle": "Regulatory challenges",
        "question": (
            "Analyze the regulatory challenges and frameworks needed to "
            "ensure safe and effective deployment of AI systems in clinical "
            "practice worldwide."
        ),
    },
]


def build_full_prompt(prompt_entry: dict) -> str:
    """Combine a prompt question with the standard output template."""
    return (
        f"{prompt_entry['question']}\n\n"
        f"{OUTPUT_TEMPLATE}"
    )


def get_all_prompts() -> list[tuple[str, str, str]]:
    """Return list of (prompt_id, angle, full_prompt_text)."""
    return [
        (p["id"], p["angle"], build_full_prompt(p))
        for p in PROMPTS
    ]
