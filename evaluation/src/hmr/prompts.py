"""System and user prompts for Bongard-LOGO classification.

The system prompt is keyed off ``bongard_context`` to switch between:
    base                        - generic instructions
    action_description          - story-style step-by-step descriptions
    action_description_concept  - story + concept hint
    aps                         - raw action-program strings
    aps_concept                 - aps + concept hint
    dvrl                        - direct visual rule learning (images only)
    dvrl_concept                - dvrl + concept hint
    free_description[_concept]  - prose descriptions of each figure (open-weight replication)
"""

from __future__ import annotations


def system_eval_prompt(bongard_context: str = "None", concept: str = "") -> str:
    """Return the system prompt for the given Bongard context.

    Note ordering: the ``*_concept`` variants must be checked before their
    bare counterparts because they share a prefix.
    """

    if bongard_context == "base":
        return _base_prompt()
    # Open-weight replication: prose descriptions written by the model itself.
    if "free_description_concept" in bongard_context:
        return _free_description_prompt(concept=concept)
    if "free_description" in bongard_context:
        return _free_description_prompt(concept=None)
    if "action_description_concept" in bongard_context:
        return _action_description_prompt(concept=concept)
    if "action_description" in bongard_context:
        return _action_description_prompt(concept=None)
    if "aps_concept" in bongard_context:
        return _aps_prompt(concept=concept)
    if "aps" in bongard_context:
        return _aps_prompt(concept=None)
    if "dvrl_concept" in bongard_context:
        return _dvrl_prompt(concept=concept)
    if "dvrl" in bongard_context:
        return _dvrl_prompt(concept=None)
    raise ValueError(f"Unknown bongard_context: {bongard_context!r}")


def user_eval_prompt(all_image_specs: list, m: int, n: int) -> str:
    """Pack positive, negative, and test specs into the user prompt (text-based)."""
    return f"""
        Here are the abstract descriptions (action programs) for the Bongard puzzle. Please analyze them according to the instructions and provide your response in the specified JSON format.

        **Positive Samples (pos):** [First {m} images]
        {_format_specs(all_image_specs[:m], 0)}

        **Negative Samples (neg):** [next {m} to {m+n} images]
        {_format_specs(all_image_specs[m:m+n], m)}

        **Final Test Sample (action program below):**
        {_format_specs([all_image_specs[-1]], m + n)}
        """


def user_eval_prompt_dvrl(m: int, n: int) -> str:
    """User prompt for DVRL: the images are passed directly to the VLM.

    The model receives m+n+1 images in order: first *m* positive samples,
    then *n* negative samples, then the test image.  The prompt references
    them by their positional index so no text descriptions are needed.
    """
    return f"""
        Here are the images for the Bongard puzzle. Analyze them according to the instructions and provide your response in the specified JSON format.

        **Positive Samples (pos):** Images 1\u2013{m}
        **Negative Samples (neg):** Images {m + 1}\u2013{m + n}
        **Test Sample:** Image {m + n + 1}
        """


def _format_specs(specs: list, count: int) -> str:
    if not specs:
        return ""
    if isinstance(specs[0], list):
        lines = []
        for i, spec_list in enumerate(specs, count + 1):
            lines.append(f"Figure {i}:\n{spec_list}\n")
        return "\n".join(lines)
    return "".join(specs)


# ---------------------------------------------------------------------------
# Prompt bodies
# ---------------------------------------------------------------------------

_JSON_KEYS = """{
            "Analysis": "A brief analysis of the distinguishing characteristics you observed in the 'pos' samples that make them distinct from 'neg' samples.",
            "Rule": "A clear and concise rule that defines the positive samples.",
            "Test Image": "A summary of the test image's properties as they relate to the derived rule.",
            "Conclusion": "Your final categorization of the test image. The value must be either 'pos' or 'neg'."
        }"""


def _base_prompt() -> str:
    return f"""
        **Your Task and Required Output:**
        Analyze the provided samples and respond *only* with a single JSON object. Do not include any additional text, explanations, or markdown formatting outside of the JSON structure.

        The JSON object must have the following keys:
        {_JSON_KEYS}
        """


def _action_description_prompt(concept: str | None) -> str:
    concept_block = (
        f"\n\n        Here is the overall concept behind the positive samples: **{concept}**\n"
        if concept else ""
    )
    return f"""
        **Objective:** Solve a Bongard-style visual reasoning puzzle by identifying the rule that distinguishes a "positive" set of programs from a "negative" set.

        **Input Context:**
        - You will receive human-readable, step-by-step instructions that describe how to construct a set of abstract shapes. These are called "action descriptions".
        - You will NOT be shown any images. Your task is to reason directly from these textual descriptions.

        **Your Core Task:**
        1.  Analyze the provided "action descriptions" for the positive, negative, and test sets.
        2.  Infer the final geometric properties of the shapes from these descriptions.
        3.  Identify the abstract rule that defines the positive set.
        4.  **Crucially, your reasoning must focus on the final geometric properties of the shapes, not the specific steps used to construct them.**
{concept_block}
        **Your Task and Required Output:**
        Respond **only** with a single JSON object. Do not include any text or explanations outside of the JSON structure.

        The JSON object must have the following keys:
        {_JSON_KEYS}
        """


def _free_description_prompt(concept: str | None) -> str:
    """As _action_description_prompt, for prose descriptions instead of step recipes."""
    return _action_description_prompt(concept).replace(
        '- You will receive human-readable, step-by-step instructions that describe how to construct a set of abstract shapes. These are called "action descriptions".',
        '- You will receive a written description of each figure in plain prose: its shapes, parts, strokes, textures and spatial arrangement. These are called "action descriptions".')


def _dvrl_prompt(concept: str | None) -> str:
    concept_block = (
        f"\n\n        Here is the overall concept behind the positive samples: **{concept}**\n"
        if concept else ""
    )
    return f"""
        **Objective:** Solve a Bongard-style visual reasoning puzzle by identifying the rule that distinguishes a "positive" set of images from a "negative" set.

        **Input Context:**
        - You will receive all puzzle images directly.  The first images are the positive set, followed by the negative set, and finally a single test image.
        - Your task is to reason about the visual properties you observe in the images.

        **Your Core Task:**
        1.  Carefully examine the positive and negative images and identify the abstract visual rule that separates them.
        2.  Apply that rule to classify the test image.
        3.  **Focus on the final geometric and visual properties of the shapes (e.g., number of sides, curvature, symmetry), not superficial pixel-level details.**
{concept_block}
        **Your Task and Required Output:**
        Respond **only** with a single JSON object. Do not include any text or explanations outside of the JSON structure.

        The JSON object must have the following keys:
        {_JSON_KEYS}
        """


def _aps_prompt(concept: str | None) -> str:
    concept_block = (
        f"\n\n        Here is the overall concept behind the positive samples: **{concept}**\n"
        if concept else ""
    )
    return f"""
        **Objective:** Solve a Bongard-style visual reasoning puzzle by identifying the rule that distinguishes a "positive" set of programs from a "negative" set.

        **Input Context: Action Program Summary (APS)**
        You will receive abstract descriptions which are the **source code** (called "action programs") used to programmatically generate the images. Understanding the technical details of this generation process is essential for solving the puzzle.

        **Action Program Details:**
        - **Structure:** The input is a nested list: `OneStrokeShape` -> `BasicAction`.
        - **Action Strings:** Each `BasicAction` is a string like `line_TYPE_LENGTH-TURNANGLE` or `arc_TYPE_ARCANGLE_ARCRADIUS-TURNANGLE`.
        - **Normalization:** All numeric values (lengths, angles) are normalized to a [0, 1] range.
        - **Stroke Style:** The stroke `TYPE` (e.g., "triangle", "zigzag") is a visual property applied during generation and is not part of the core geometry.

        **Your Core Task:**
        1.  Analyze the structure and values within the `pos` and `neg` action programs.
        2.  Identify the abstract rule that defines the positive set. This rule might relate to stroke types, counts of actions, geometric properties implied by the normalized values, etc.
        3.  **Crucially, your reasoning must focus on the final geometric properties of the shapes that these programs would generate, not the specific construction steps.**
{concept_block}
        **Your Task and Required Output:**
        Respond **only** with a single JSON object. Do not include any text or explanations outside of the JSON structure.

        The JSON object must have the following keys:
        {_JSON_KEYS}
        """
