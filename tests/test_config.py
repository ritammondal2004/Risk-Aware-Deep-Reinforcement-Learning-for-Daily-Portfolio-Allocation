import copy       
import pytest

from src.config import CFG, MODEL_SPECS, assert_controlled_ablation


def test_ablation_changes_one_mechanism_per_step():
    assert assert_controlled_ablation()

  
def test_ablation_check_catches_a_confound():
    bad = copy.deepcopy(MODEL_SPECS)
    bad["M2"]["reward"] = "dsr"          # M1->M2 now changes encoder AND reward
    with pytest.raises(AssertionError):
        assert_controlled_ablation(bad)


def test_dimensions():
    N = CFG.n_assets  
    assert CFG.feature_dim == 13 * N and CFG.state_dim == 14 * N + 1 and CFG.action_dim == N + 1  
             