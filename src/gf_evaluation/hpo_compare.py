"""Pure exploratory comparison over a complete declared training-seed product."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from gf_common.domain import ContractError, Json, integer
from gf_evaluation.hpo import EvaluationPlan, SeedEvaluationProjection, aggregate_evaluation


@dataclass(frozen=True)
class ComparisonPanel:
    left_training_seed: int
    right_training_seed: int
    right_run_id: str
    right_inference_sha256: str
    right_source_receipt_sha256: str
    right_decision_budget: int
    left: SeedEvaluationProjection


def comparison_table(left_seeds: tuple[int, ...], right_seeds: tuple[int, ...], game_seeds: tuple[str, ...],
                     decision_budget: int, conditions_sha256: str, panels: tuple[ComparisonPanel, ...]) -> Json:
    """Inputs are independently verified projections; this creates no authority."""
    for seeds in (left_seeds, right_seeds):
        if type(seeds) is not tuple or not seeds or len(set(seeds)) != len(seeds):
            raise ContractError('complete unique declared comparison seed sets required')
        for seed in seeds:
            integer(seed, 0, 2**63 - 1)
    integer(decision_budget, 1)
    expected = {(left, right) for left in left_seeds for right in right_seeds}
    if type(panels) is not tuple or len(panels) != len(expected):
        raise ContractError('comparison requires the full declared seed-pair product')
    rows: dict[tuple[int, int], Json] = {}
    game_ids: set[str] = set()
    for panel in panels:
        if type(panel) is not ComparisonPanel:
            raise ContractError('typed comparison panel required')
        integer(panel.left_training_seed, 0, 2**63 - 1)
        integer(panel.right_training_seed, 0, 2**63 - 1)
        key = (panel.left_training_seed, panel.right_training_seed)
        if key not in expected or key in rows or panel.left.run.training_seed != panel.left_training_seed:
            raise ContractError('extra/duplicate/substituted comparison seed pair')
        integer(panel.right_decision_budget, 1)
        if panel.right_decision_budget != decision_budget or panel.left.conditions_sha256 != conditions_sha256:
            raise ContractError('comparison must use common exact D and conditions')
        if type(panel.right_run_id) is not str or not panel.right_run_id:
            raise ContractError('actual right parent identity required')
        for value in (panel.right_inference_sha256, panel.right_source_receipt_sha256):
            if type(value) is not str or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
                raise ContractError('right parent comparison identity requires SHA256')
        plan = EvaluationPlan((panel.left_training_seed,), game_seeds, decision_budget, conditions_sha256, (panel.left.run,))
        score = aggregate_evaluation(plan, (panel.left,))
        for game in panel.left.games:
            if game.identity.game_id in game_ids:
                raise ContractError('comparison cannot reuse an actual GameID across seed pairs')
            game_ids.add(game.identity.game_id)
        rows[key] = {'left_training_seed': panel.left_training_seed, 'right_training_seed': panel.right_training_seed,
            'left_run_id': panel.left.run.run_id, 'right_run_id': panel.right_run_id,
            'left_inference_sha256': panel.left.run.checkpoint_sha256, 'right_inference_sha256': panel.right_inference_sha256,
            'left_source_receipt_sha256': panel.left.run.source_receipt_sha256,
            'right_source_receipt_sha256': panel.right_source_receipt_sha256,
            'paired_score': score.value, 'games': [asdict(game) for game in panel.left.games]}
    left_means = [sum(rows[(left, right)]['paired_score'] for right in right_seeds) / len(right_seeds) for left in left_seeds]
    return {'schema_version': 1, 'kind': 'hpo-exploratory-comparison', 'label': 'exploratory-hpo-comparison',
        'score_recipe': 'hpo-comparison-equal-seed-product-v1', 'decision_budget': decision_budget,
        'conditions_sha256': conditions_sha256, 'left_training_seeds': list(left_seeds), 'right_training_seeds': list(right_seeds),
        'game_seeds': list(game_seeds), 'matchups': [rows[(left, right)] for left in left_seeds for right in right_seeds],
        'left_paired_score': sum(left_means) / len(left_means)}
