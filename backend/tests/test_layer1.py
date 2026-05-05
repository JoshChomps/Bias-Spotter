"""
Layer 1 Tests — Lexicon matching accuracy and coverage.
"""

import pytest

from app.pipeline.layer1_lexicon import Layer1Matcher


@pytest.fixture
def matcher() -> Layer1Matcher:
    """Create a matcher with the default lexicon."""
    return Layer1Matcher()


class TestLayer1Matching:
    """Test that the lexicon catches expected patterns."""

    def test_false_scarcity_limited_time(self, matcher: Layer1Matcher) -> None:
        text = "This is a limited time offer, act now!"
        matches = matcher.match(text)
        techniques = {m.technique for m in matches}
        assert "false_scarcity" in techniques

    def test_appeal_to_authority(self, matcher: Layer1Matcher) -> None:
        text = "Experts say that this product is the best on the market."
        matches = matcher.match(text)
        techniques = {m.technique for m in matches}
        assert "appeal_to_authority" in techniques

    def test_bandwagon(self, matcher: Layer1Matcher) -> None:
        text = "Millions of people have already signed up. Everyone agrees this is important."
        matches = matcher.match(text)
        techniques = {m.technique for m in matches}
        assert "bandwagon" in techniques

    def test_loaded_language(self, matcher: Layer1Matcher) -> None:
        text = "The brutal regime continues its devastating assault on freedom."
        matches = matcher.match(text)
        techniques = {m.technique for m in matches}
        assert "loaded_language" in techniques

    def test_black_and_white(self, matcher: Layer1Matcher) -> None:
        text = "You're either with us or against us."
        matches = matcher.match(text)
        techniques = {m.technique for m in matches}
        assert "black_and_white" in techniques

    def test_clean_text_no_matches(self, matcher: Layer1Matcher) -> None:
        text = "The weather forecast predicts rain tomorrow afternoon."
        matches = matcher.match(text)
        assert len(matches) == 0, f"Clean text should have no matches, got: {matches}"

    def test_match_offsets_are_correct(self, matcher: Layer1Matcher) -> None:
        text = "Act now before it's too late!"
        matches = matcher.match(text)
        for match in matches:
            assert text[match.start : match.end] == match.text_span

    def test_multiple_techniques_in_one_text(self, matcher: Layer1Matcher) -> None:
        text = (
            "Experts recommend this limited time offer. "
            "Millions of customers can't be wrong. Act now!"
        )
        matches = matcher.match(text)
        techniques = {m.technique for m in matches}
        assert len(techniques) >= 2, "Should detect multiple different techniques."

    def test_thought_terminating_cliche(self, matcher: Layer1Matcher) -> None:
        text = "Well, it is what it is. End of story."
        matches = matcher.match(text)
        techniques = {m.technique for m in matches}
        assert "thought_terminating_cliches" in techniques
