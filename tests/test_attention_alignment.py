# tests/test_attention_alignment.py
from feedflipnets.eval.alignment import attention_block_theta, per_path_theta, theta_deg


def test_attention_block_theta_is_worst_matrix():
    cos = {"Wq": 0.9, "Wk": 0.8, "Wv": 0.2, "Wo": 1.0, "W1": 0.99, "W2": 1.0}
    # worst over the FOUR attention matrices only (MLP excluded)
    assert abs(attention_block_theta(cos) - theta_deg(0.2)) < 1e-9


def test_per_path_theta_splits_value_and_score():
    cos = {"Wq": 0.5, "Wk": 0.4, "Wv": 0.9, "Wo": 1.0}
    paths = per_path_theta(cos)
    assert abs(paths["value"] - theta_deg(0.9)) < 1e-9  # min over {Wv, Wo} = worst = Wv
    assert abs(paths["score"] - theta_deg(0.4)) < 1e-9  # min over {Wq, Wk} = worst = Wk
