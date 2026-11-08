"""Exact synthetic participant journal recipe; no collected test dependencies."""
import uuid

from gf_common.actions import PublicCandidateGenerator
from gf_common.domain import ActionConfig, PolicyDecision


def final_state(value, outcome):
    return dict(value.observation, result=outcome, phase="terminal", actionable=False,
                typed_action_schema=[], current_player=None, required_choice=None)


def journal_decision(journal, value, command_id=None):
    command_id = str(uuid.uuid4())
    candidates = PublicCandidateGenerator(ActionConfig()).generate(value)
    selection = PolicyDecision(0, "random-v1", -1.0)
    command = {"command_id": command_id, "expected_revision": value.observation["state_revision"], "action": candidates.actions[0]}
    index = journal.begin(value, candidates, selection, command, 1.5)
    acknowledgement = {"command_id": command_id, "previous_revision": value.observation["state_revision"], "state_revision": value.observation["state_revision"] + 1, "status": "active", "result": None}
    journal.acknowledge(index, acknowledgement)
    return index
