"""Zero-decision episodes preserve reward checks and numeric output shape."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gf_common.domain import ContractError
from gf_common.preprocessing import PreparationConfig
from gf_models.config import EncoderConfig
from gf_models.encoder import feature_names
from gf_training.native_preparation import NativePreparedInputs


@pytest.mark.parametrize('bad_empty_reward', [False, True])
def test_fixed_writer_checks_empty_episode_but_writes_only_decisions(tmp_path, monkeypatch, bad_empty_reward):
    config = EncoderConfig()
    width = len(feature_names(config))

    def readonly(value, dtype):
        result = np.asarray(value, dtype=dtype)
        result.flags.writeable = False
        return result

    batch = SimpleNamespace(features=readonly(np.ones((2, width)), np.float64),
                            state_features=readonly(np.zeros((2, width)), np.float64),
                            selected=readonly([0, 0], np.int64),
                            candidate_counts=readonly([1, 1], np.int64),
                            episode_offsets=readonly([0, 0, 1, 2], np.int64),
                            candidate_offsets=readonly([0, 1, 2], np.int64),
                            old_logprob=readonly([0, 0], np.float64),
                            old_value=readonly([0, 0], np.float64),
                            old_logprob_present=readonly([0, 0], np.int64),
                            old_value_present=readonly([0, 0], np.int64),
                            winner=readonly([0, 1, 2], np.int64),
                            learner_viewer=readonly([1, 1, 2], np.int64),
                            split=readonly([0, 0, 1], np.int64),
                            terminal_rewards=readonly([[int(bad_empty_reward), 0], [1, -1], [-1, 1]], np.float64),
                            feature_layout='selected')

    class Lease:
        def release(self):
            pass

    class Owner:
        def __init__(self):
            self.diagnostics = {}
            self.ended = False

        def next_batch(self, *_):
            if self.ended:
                return None
            self.ended = True
            return batch

        def proof(self):
            return {'structural_valid': True, 'sample_identity_sha256': '0' * 64,
                    'dataset_manifest': {'train_decisions': 1, 'validation_decisions': 1}}

        def reserve_staging(self, *_):
            return Lease()

        def check(self):
            pass

        def finalize(self):
            pass

        def abort(self):
            pass

    monkeypatch.setattr('gf_training.native_preparation._owner', lambda *_: Owner())
    settings = replace(PreparationConfig(), backend='rust')
    if bad_empty_reward:
        with pytest.raises(ContractError, match='both-view'):
            NativePreparedInputs(tmp_path / 'unused.json', config, settings, deadline=float('inf'))
    else:
        with NativePreparedInputs(tmp_path / 'unused.json', config, settings, deadline=float('inf')) as prepared:
            wall = prepared.diagnostics['native_prepared_inputs_wall']
            assert wall['success'] is True
            assert wall['parent_nanoseconds'] == sum(wall['partition_nanoseconds'].values())
            assert wall['residual_nanoseconds'] == 0
            construction_parent = wall['parent_nanoseconds']
            prepared.write_numeric(tmp_path / 'numeric.npz')
        assert prepared.diagnostics['native_prepared_inputs_wall']['parent_nanoseconds'] == construction_parent
        with np.load(tmp_path / 'numeric.npz', allow_pickle=False) as result:
            np.testing.assert_array_equal(result['x'], np.ones((1, width)))
            np.testing.assert_array_equal(result['validation_x'], np.ones((1, width)))
            np.testing.assert_array_equal(result['y'], [1])
            np.testing.assert_array_equal(result['validation_y'], [1])
