from euregsearch.qa.esma import MINIMUM_PAIRS, QAEntry, check_abort_gate, extract_entries

PAGE = """
Question 1 [Last update: 12/05/2023]
Does the KID obligation apply to professional clients?

Answer 1
No. Article 2 of Regulation (EU) No 1286/2014 limits the obligation to retail investors.

Question 2
When does an appropriateness assessment apply?

Answer 2
See Article 25 of Directive 2014/65/EU.
"""


def entries():
    return extract_entries([PAGE], language="en", source="JC_2023_22")


def test_extracts_both_pairs():
    assert len(entries()) == 2


def test_question_text_is_captured():
    assert "professional clients" in entries()[0].question


def test_citations_are_resolved_from_the_answer():
    assert entries()[0].citations == [("32014R1286", "2")]
    assert entries()[1].citations == [("32014L0065", "25")]


def test_source_and_language_are_recorded():
    entry = entries()[0]
    assert entry.source == "JC_2023_22"
    assert entry.language == "en"


def test_entries_without_citations_are_dropped():
    page = "Question 1\nIs this in scope?\n\nAnswer 1\nYes, generally."
    assert extract_entries([page], language="en", source="X") == []


def test_abort_gate_fails_below_the_floor():
    few = [QAEntry(question="q", answer="a", language="en", source="s",
                   citations=[("32014L0065", "25")])] * 10
    count, passed = check_abort_gate(few)
    assert count == 10 and passed is False


def test_abort_gate_passes_at_the_floor():
    many = [QAEntry(question="q", answer="a", language="en", source="s",
                    citations=[("32014L0065", "25")])] * MINIMUM_PAIRS
    count, passed = check_abort_gate(many)
    assert count == MINIMUM_PAIRS and passed is True


def test_minimum_is_the_value_fixed_in_the_spec():
    assert MINIMUM_PAIRS == 120
