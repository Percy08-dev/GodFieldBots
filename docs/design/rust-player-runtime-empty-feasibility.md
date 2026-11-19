# Reachability of the healthy-empty HPO engine case

Status: accepted conditional source-based verification disposition.
Independent architecture review READY on proposal SHA
429df7f1164b156d4bb4796e489912d744c775da8bc406c321125bc8d5ebb36a.
Owner: root. No production behavior, schedule, selection, engine or data format
change. This addresses the explicitly unrun genuine empty-capture experiment,
not the separate required genuine pruned-checkpoint resume/ordinary-HPO acceptance.

## Pinned source and proof

Read-only bot_verifier /root/empty_capture_feasibility inspected Bots baseline
 a83288b and the accepted empty implementation at0488, plus Simulator
072dabbca1458c9526897781d2fd64edeeb4f9ca at /tmp/gf-issue72-shared-inference.
No tests, builds, processes, games or seed search were launched. Root separately
checked the native actor/inspection paths below.

1. HPO config admits collection_games from8 through65536 in multiples of8
   (gf_training/hpo_config.py:46; hpo_collection.py:22–44). One-game collections
   are not admitted. Every full schedule consists of paired TWO-player episodes
   with the learner seat swapped. Mixed Refresh preserves that pair structure.
2. Both episodes use the SAME group game seed. The collector starts the group
   server with that seed and its episode-count reservation limit
   (rust_player_collection.py:214; simulator_client/process.py:80). Pinned
   simulator server/application/mod.rs:421 returns the repeated seed unchanged.
3. Pinned GameEngine::new deterministically selects player1 orplayer2 from that
   seed, sets turn1/AwaitingAction/current_player/TurnAction and no result, then
   deals the normal initial hands (engine/resolution/entry.rs:42–88). Normative
   specs03:863 and09:105 agree. Policy labels/learner assignment do not enter that
   engine initialization. The paired episodes therefore have the same first seat.
4. The learner occupies that first seat in ONE episode of each pair. A complete
   natural episode cannot finish without that required first action. Native MLP
   OwnPolicy::select returns None unconditionally, so the actor follows its MLP
   inference route even for a singleton candidate table (game.rs:188–195).
   After actual Act ACK the actor records the action and increments the seat's
   decision count (game.rs:500–522); forced single choices are not omitted. The
   storage adapter begins/ACKs that frame (rust_player_storage.py:470–483), journal
   parsing accepts it (structured_journal.py:535–541), and complete Raw/Replay
   inspection places accepted decisions in EpisodeRecords.frames
   (structured_source.py:665–683).
5. Complete Raw/Replay inspection chooses the configured learner viewer without
   filtering a natural subset (rust_player_source.py:268–274,646–674). HPO counts
   are len(record.frames) for every selected own-seat record. Each pair contributes
   at least ONE collected learner frame. Thus L>=collection_games/2>=4.
6. The common loop stops when its prescribed decision budget is exhausted BEFORE
   another collection (ppo_run.py:686–692 and RustHpoOperations.stop:1063–1065). For a dispatched cohort remainingD>0;
   selection_for_counts takes min(remainingD,L). Therefore selected frames>0.
   The same argument holds for the first and every later dispatched cohort.

This is conditional on the current successful complete cohort and recorded
schedule/seed/actor contracts. Truncation, abort, unsupported rules, transport or
capture failure cannot be reclassified as a healthy empty natural collection.
This proof does not claim seed-only deterministic training or performance.

## Proposed verification disposition

For these currently admitted paired/repeated-seed schedules (including8,16,32),
a genuine all-natural zero-own-seat engine capture is UNREACHABLE. Do not launch
an open seed search or alter the learner seats, budget, policy, repeated seed,
cohort size after observing decisions, engine state or outcome merely to create
it. No bounded truthful live setup exists under these contracts.

Keep the implemented generic healthy-empty compatibility behavior and focused
DATA/ownership/cold-marker/extra-Replay negative tests. Label the real engine
zero-frame experiment NOT APPLICABLE under the reviewed invariants, rather than
PASS or an indefinitely outstanding successful run. Existing synthetic DATA
never becomes Source/Parent authority or real-engine evidence. The separate
actual full pruned-checkpoint resume and all other approved Task6/7 conditions
remain mandatory and unchanged.

Revisit this proof and add a genuine first/later empty-engine case BEFORE adopting
any change to the paired learner-seat schedule, repeated seed reservation,
engine first-action semantics, singleton/action-record counting, learner-viewer
selection, precollection budget stop, or a new producer/curriculum that permits
zero frames. This is a verification-domain condition, not waived implementation
correctness or accepted permanent debt. An observed zero under current valid
contracts would itself be an invariant violation requiring investigation.
