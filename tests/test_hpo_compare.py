"""Complete seed-product comparison is exploratory and separate from Optuna."""
from dataclasses import replace

import pytest

from gf_common.domain import ContractError
from gf_evaluation.hpo import (
    GameIdentity,
    GameProjection,
    RunIdentity,
    SeedEvaluationProjection,
)
from gf_evaluation.hpo_compare import ComparisonPanel, comparison_table


def panels():
    result = []
    for left in (11,22):
        for right in (3,4,5):
            identities = tuple(GameIdentity("game-seed",seat,f"episode-{left}-{right}-{seat}",f"game-{left}-{right}-{seat}") for seat in (1,2))
            run = RunIdentity(left,f"left-{left}","a"*64,"b"*64,identities)
            # First left seed always wins, second always loses: equal .5.
            games = tuple(GameProjection.from_result(identity,result={"win":{"winner":identity.learner_seat if left==11 else 3-identity.learner_seat}}) for identity in identities)
            projection = SeedEvaluationProjection(run,100,"c"*64,games)
            result.append(ComparisonPanel(left,right,f"right-{right}","d"*64,"e"*64,100,projection))
    return tuple(result)


def test_full_unequal_seed_sets_use_every_declared_pair_without_relabeling():
    table = comparison_table((11,22),(3,4,5),("game-seed",),100,"c"*64,panels())
    assert len(table["matchups"]) == 6
    assert table["left_paired_score"] == .5
    assert table["label"] == "exploratory-hpo-comparison"
    assert {row["right_training_seed"] for row in table["matchups"]} == {3,4,5}
    assert "value" not in table


@pytest.mark.parametrize("fault", ["missing","duplicate","extra","budget","conditions","failed","duplicate-game"])
def test_comparison_refuses_partial_failed_or_substituted_panels(fault):
    rows = panels()
    if fault == "missing": rows = rows[:-1]
    elif fault == "duplicate": rows = rows[:-1]+(rows[0],)
    elif fault == "extra": rows += (replace(rows[0],right_training_seed=6),)
    elif fault == "budget": rows = (replace(rows[0],right_decision_budget=99),*rows[1:])
    elif fault == "conditions": rows = (replace(rows[0],left=replace(rows[0].left,conditions_sha256="f"*64)),*rows[1:])
    elif fault == "failed": rows = (replace(rows[0],left=replace(rows[0].left,cleanup_ok=False)),*rows[1:])
    else:
        duplicate = rows[0].left.run.games[0].game_id
        last = rows[-1]
        identity = replace(last.left.run.games[0],game_id=duplicate)
        run = replace(last.left.run,games=(identity,*last.left.run.games[1:]))
        game = replace(last.left.games[0],identity=identity)
        rows = (*rows[:-1],replace(last,left=replace(last.left,run=run,games=(game,*last.left.games[1:]))))
    with pytest.raises(ContractError): comparison_table((11,22),(3,4,5),("game-seed",),100,"c"*64,rows)
