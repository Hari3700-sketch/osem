import groq_client


def test_detects_incomplete_sentence():
    assert groq_client._looks_truncated("This is a complete answer.") is False
    assert groq_client._looks_truncated(
        "This is a detailed answer that explains the process, eligibility requirements, and the official steps to apply for the scheme before the deadline"
    ) is True


def test_uses_higher_token_budget_for_long_answers():
    assert groq_client._choose_max_tokens("Explain the full scheme details with eligibility and process", "en") >= 1600
    assert 800 <= groq_client._choose_max_tokens("What is MSME?", "en") <= 1200
