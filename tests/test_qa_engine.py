from src.modules.qa_engine import interpret_prediction, possible_factors

def test_low_confidence_requires_manual_review():
    result = interpret_prediction("defective", 0.4)
    assert result["status"].startswith("UNCERTAIN")

def test_possible_factors_are_not_empty_for_known_defect():
    assert possible_factors("chipping")
