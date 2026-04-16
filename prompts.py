"""
Multi-topic prompts for the LLM Stochasticity Exploration Experiment.

Each topic has 10–20 prompts covering different angles.
Every prompt shares a single OUTPUT_TEMPLATE so responses are structurally
comparable across models, repetitions, and topics.
"""

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



# ---------------------------------------------------------------------------
# 10 prompts — Climate Change & Environmental Policy
# ---------------------------------------------------------------------------
CLIMATE_PROMPTS: list[dict[str, str]] = [
    {
        "id": "C01",
        "angle": "General overview",
        "question": (
            "Explain the current scientific understanding of climate change, "
            "its primary causes, and the projected impacts on global ecosystems."
        ),
    },
    {
        "id": "C02",
        "angle": "Renewable energy transition",
        "question": (
            "Analyze the challenges and opportunities in transitioning from "
            "fossil fuels to renewable energy sources on a global scale."
        ),
    },
    {
        "id": "C03",
        "angle": "Carbon capture",
        "question": (
            "Discuss the viability and scalability of carbon capture and "
            "storage technologies as a strategy to mitigate climate change."
        ),
    },
    {
        "id": "C04",
        "angle": "Agriculture and food security",
        "question": (
            "Evaluate how climate change is affecting global agriculture, "
            "food supply chains, and strategies for ensuring food security."
        ),
    },
    {
        "id": "C05",
        "angle": "Biodiversity loss",
        "question": (
            "Describe the relationship between climate change and biodiversity "
            "loss, including impacts on marine and terrestrial species."
        ),
    },
    {
        "id": "C06",
        "angle": "Economic impact",
        "question": (
            "Assess the economic costs of climate change, including damage to "
            "infrastructure, healthcare burdens, and impacts on labor productivity."
        ),
    },
    {
        "id": "C07",
        "angle": "Policy frameworks",
        "question": (
            "Analyze the effectiveness of international climate agreements such "
            "as the Paris Agreement in driving meaningful emission reductions."
        ),
    },
    {
        "id": "C08",
        "angle": "Urban resilience",
        "question": (
            "Evaluate how cities can build resilience against climate change "
            "through urban planning, green infrastructure, and adaptation strategies."
        ),
    },
    {
        "id": "C09",
        "angle": "Climate justice",
        "question": (
            "Discuss the concept of climate justice and how the impacts of "
            "climate change disproportionately affect vulnerable and developing nations."
        ),
    },
    {
        "id": "C10",
        "angle": "Water resources",
        "question": (
            "Explain how climate change is altering global water cycles, "
            "threatening freshwater availability, and increasing flood and drought risks."
        ),
    },
]

# ---------------------------------------------------------------------------
# 10 prompts — Software Engineering & Technology
# ---------------------------------------------------------------------------
SOFTWARE_PROMPTS: list[dict[str, str]] = [
    {
        "id": "S01",
        "angle": "General overview",
        "question": (
            "Explain the current state of modern software engineering practices "
            "and how they have evolved to meet the demands of large-scale systems."
        ),
    },
    {
        "id": "S02",
        "angle": "DevOps and CI/CD",
        "question": (
            "Analyze how DevOps practices and continuous integration/continuous "
            "deployment pipelines improve software delivery speed and reliability."
        ),
    },
    {
        "id": "S03",
        "angle": "Cloud computing",
        "question": (
            "Discuss the impact of cloud computing on software architecture, "
            "scalability, and the economics of running production systems."
        ),
    },
    {
        "id": "S04",
        "angle": "Cybersecurity",
        "question": (
            "Evaluate the growing challenges of cybersecurity in software "
            "development and strategies for building secure applications."
        ),
    },
    {
        "id": "S05",
        "angle": "Open source",
        "question": (
            "Describe the role of open-source software in the modern technology "
            "ecosystem and its impact on innovation and collaboration."
        ),
    },
    {
        "id": "S06",
        "angle": "Technical debt",
        "question": (
            "Assess the causes and consequences of technical debt in software "
            "projects and effective strategies for managing and reducing it."
        ),
    },
    {
        "id": "S07",
        "angle": "Microservices architecture",
        "question": (
            "Analyze the trade-offs between microservices and monolithic "
            "architectures for building and maintaining complex software systems."
        ),
    },
    {
        "id": "S08",
        "angle": "Testing and quality",
        "question": (
            "Evaluate modern approaches to software testing, including automated "
            "testing, property-based testing, and their impact on software quality."
        ),
    },
    {
        "id": "S09",
        "angle": "AI-assisted development",
        "question": (
            "Discuss how artificial intelligence tools are transforming software "
            "development workflows, including code generation, review, and debugging."
        ),
    },
    {
        "id": "S10",
        "angle": "Ethics in technology",
        "question": (
            "Analyze the ethical responsibilities of software engineers in "
            "designing systems that affect privacy, fairness, and societal well-being."
        ),
    },
]

# ---------------------------------------------------------------------------
# Topic registry — maps topic names to their prompt lists
# ---------------------------------------------------------------------------
TOPIC_PROMPTS = {
    "healthcare": PROMPTS,
    "climate": CLIMATE_PROMPTS,
    "software": SOFTWARE_PROMPTS,
}

AVAILABLE_TOPICS = list(TOPIC_PROMPTS.keys())


def build_full_prompt(prompt_entry: dict) -> str:
    """Combine a prompt question with the standard output template."""
    return (
        f"{prompt_entry['question']}\n\n"
        f"{OUTPUT_TEMPLATE}"
    )


def get_all_prompts(topic: str = "healthcare") -> list[tuple[str, str, str]]:
    """
    Return list of (prompt_id, angle, full_prompt_text).

    Args:
        topic: One of "healthcare", "climate", "software".
               Defaults to "healthcare" for backward compatibility.
    """
    prompts = TOPIC_PROMPTS.get(topic, PROMPTS)
    return [
        (p["id"], p["angle"], build_full_prompt(p))
        for p in prompts
    ]
