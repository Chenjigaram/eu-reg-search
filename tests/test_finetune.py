from euregsearch.train.finetune import DEFAULT_DIMENSIONS, TrainConfig, to_examples

PAIRS = [
    ("query one", "passage one", ("32014L0065", "25")),
    ("query two", "passage two", ("32014R1286", "8")),
]


def test_examples_carry_anchor_and_positive():
    examples = to_examples(PAIRS)
    assert examples[0]["anchor"] == "query: query one"
    assert examples[0]["positive"] == "passage: passage one"


def test_one_example_per_pair():
    assert len(to_examples(PAIRS)) == 2


def test_empty_pairs_give_no_examples():
    assert to_examples([]) == []


def test_default_dimensions_are_descending_and_match_the_spec():
    assert DEFAULT_DIMENSIONS == (384, 256, 128, 64)
    assert list(DEFAULT_DIMENSIONS) == sorted(DEFAULT_DIMENSIONS, reverse=True)


def test_config_defaults_are_cpu_safe():
    config = TrainConfig()
    assert config.batch_size <= 32
    assert config.model_name == "intfloat/multilingual-e5-small"
