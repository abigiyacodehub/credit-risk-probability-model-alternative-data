from src.predict import probability_to_risk_tier


def test_probability_to_risk_tier_boundaries():
    assert probability_to_risk_tier(0.1) == "LOW"
    assert probability_to_risk_tier(0.3) == "MEDIUM"
    assert probability_to_risk_tier(0.6) == "HIGH"
    assert probability_to_risk_tier(0.9) == "VERY_HIGH"
