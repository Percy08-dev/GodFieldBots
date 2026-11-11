"""Storage command routing only, without runtime, Source, or fit execution."""
import time

from gf_cli.card_instance import _game_tree_data_config
from gf_cli.config import load_config, require_storage_command
from gf_models.structured_tree_config import ConfiguredTreeModelV3
from tests.test_card_instance_game_tree_cli import config_file


def test_dataset_and_inference_storage_routes_reach_existing_checked_consumers(tmp_path):
    source = config_file(tmp_path)
    source.write_text(source.read_text()+'\n[storage]\ndb_root="db"\nram_root="ram"\n')
    loaded = load_config(source)
    model, _, deadline = _game_tree_data_config(loaded, time.monotonic())
    assert type(model) is ConfiguredTreeModelV3 and deadline > time.monotonic()
    require_storage_command(loaded.config, 'collect')
    assert not (tmp_path/'db').exists() and not (tmp_path/'ram').exists()
