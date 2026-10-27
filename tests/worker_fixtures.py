"""Factory imports themselves must see only sanitized runtime globals."""
import os
import sys
import time

from gf_common.domain import PolicyDecision

# Parent imports before adding its sentinel control values. The fresh worker
# module import must never see these values, even before calling its factory.
assert "GFB_PRIVATE_SENTINEL" not in os.environ
assert all("secret-game-seed-sentinel" not in a and "secret-server-url-sentinel" not in a for a in sys.argv)
print("module import output must not corrupt protocol")


class SlowPolicy:
    def reset(self, sampling_seed):
        pass

    def select(self, value, candidates):
        time.sleep(10)
        return PolicyDecision(0, "slow-v1")


def slow_factory(model_id):
    return SlowPolicy()


class StuckResetPolicy(SlowPolicy):
    def reset(self, sampling_seed):
        time.sleep(10)


def stuck_reset_factory(model_id):
    return StuckResetPolicy()


class AuditPolicy:
    def reset(self, sampling_seed):
        assert sys.argv == ["gf-policy-worker"]
        assert "GFB_PRIVATE_SENTINEL" not in os.environ
        print("reset output must not corrupt protocol")

    def select(self, value, candidates):
        assert sys.argv == ["gf-policy-worker"]
        assert "GFB_PRIVATE_SENTINEL" not in os.environ
        print("select output must not corrupt protocol")
        return PolicyDecision(0, "audit-v1")


def audit_factory(model_id):
    assert sys.argv == ["gf-policy-worker"]
    assert "GFB_PRIVATE_SENTINEL" not in os.environ
    print("factory output must not corrupt protocol")
    return AuditPolicy()


class ResistantResetPolicy(SlowPolicy):
    def reset(self, sampling_seed):
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(10)


def resistant_reset_factory(model_id):
    return ResistantResetPolicy()


def error_factory(model_id):
    raise RuntimeError("intentional worker fixture failure")


def malformed_frame_factory(model_id):
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.write(1, b"badframe")
    time.sleep(10)
    return SlowPolicy()


class ResistantReadyPolicy(AuditPolicy):
    def reset(self, sampling_seed):
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        super().reset(sampling_seed)


def resistant_ready_factory(model_id):
    return ResistantReadyPolicy()


class FixedBuyChoicePolicy(AuditPolicy):
    """Controlled test choices only; no saved prefix or experiment controls."""
    def __init__(self, accept):
        self.accept = accept

    def select(self, value, candidates):
        if value.observation["required_choice"] == "defense":
            wanted = {"type": "respond", "ordered_defenses": []}
        else:
            assert value.observation["phase"] == "resolving"
            assert value.observation["required_choice"] == "buy_accept_reject"
            wanted = {"type": "buy_decision", "accept": self.accept}
        return PolicyDecision(candidates.actions.index(wanted), "fixed-buy-component-v1")


def fixed_buy_choice_factory(model_id):
    assert model_id in ("accept", "reject")
    return FixedBuyChoicePolicy(model_id == "accept")
