from euregsearch.evaluation.report import comparison_table, slice_table

SUMMARIES = [
    {"system": "bm25", "queries": 120, "ndcg": 0.31, "recall": 0.44, "mrr": 0.28,
     "provenance_complete": True,
     "slices": {"same_language": {"queries": 60, "ndcg": 0.5, "recall": 0.6, "mrr": 0.45},
                "cross_lingual": {"queries": 60, "ndcg": 0.12, "recall": 0.2, "mrr": 0.1}}},
    {"system": "finetuned", "queries": 120, "ndcg": 0.68, "recall": 0.81, "mrr": 0.64,
     "provenance_complete": True, "slices": {}},
]


def test_comparison_lists_every_system():
    table = comparison_table(SUMMARIES)
    assert "bm25" in table and "finetuned" in table


def test_comparison_is_sorted_best_first():
    table = comparison_table(SUMMARIES)
    assert table.index("finetuned") < table.index("bm25")


def test_comparison_is_markdown():
    assert comparison_table(SUMMARIES).startswith("| System")


def test_incomplete_provenance_is_flagged_as_a_defect():
    broken = [{**SUMMARIES[0], "provenance_complete": False}]
    assert "DEFECT" in comparison_table(broken)


def test_slice_table_lists_each_slice():
    table = slice_table(SUMMARIES[0])
    assert "same_language" in table and "cross_lingual" in table


def test_slice_table_of_a_system_without_slices():
    assert "no slices" in slice_table(SUMMARIES[1]).lower()


def test_empty_summaries():
    assert "no results" in comparison_table([]).lower()
