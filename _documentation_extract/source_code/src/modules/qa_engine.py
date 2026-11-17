"""Conservative QA wording; visual predictions are not root-cause diagnoses."""
POSSIBLE_FACTORS = {
    "chipping": ["weak tablet edges", "binding/formulation properties",
                 "handling or ejection conditions"],
    "cracking": ["mechanical stress", "formulation properties",
                 "drying or compression conditions"],
    "capping": ["air entrapment", "inadequate inter-particulate bonding",
                "formulation or compression conditions"],
    "contamination": ["material contamination", "equipment cleanliness",
                      "handling conditions"],
}

def interpret_prediction(label: str, confidence: float, review_threshold: float = 0.70):
    if confidence < review_threshold:
        return {
            "status": "UNCERTAIN — MANUAL REVIEW",
            "message": "The model result is uncertain. Do not treat this sample as passed."
        }
    if label == "normal":
        return {
            "status": "NO VISIBLE DEFECT DETECTED",
            "message": "No visible defect was identified by the model. Human review remains necessary."
        }
    return {
        "status": "ATTENTION — MANUAL VERIFICATION",
        "message": f"Possible visible abnormality: {label}. Isolate for qualified QA examination."
    }

def possible_factors(defect_type: str):
    """Returns investigation prompts, never confirmed manufacturing causes."""
    return POSSIBLE_FACTORS.get(defect_type.lower(), [])


def interpret_probability(p_defective: float, low: float = 0.10, high: float = 0.90):
    """Three-state decision from P(defective). `low`/`high` are PLACEHOLDERS.

    Tune them on validation data (never the external test set) so that the defective-recall /
    false-pass gate you pre-registered is met. Only a very low P(defective) counts as
    'no visible defect'; everything in between is manual review, never an automatic pass.
    """
    if not 0.0 <= p_defective <= 1.0:
        raise ValueError("p_defective must be between 0 and 1")
    if low >= high:
        raise ValueError("low threshold must be smaller than high threshold")
    if p_defective >= high:
        return {"status": "ATTENTION — MANUAL VERIFICATION",
                "message": "Possible visible abnormality. Isolate for qualified QA examination."}
    if p_defective <= low:
        return {"status": "NO VISIBLE DEFECT DETECTED",
                "message": "No visible defect was identified by the model. Human review remains necessary."}
    return {"status": "UNCERTAIN — MANUAL REVIEW",
            "message": "The model result is uncertain. Do not treat this sample as passed."}
