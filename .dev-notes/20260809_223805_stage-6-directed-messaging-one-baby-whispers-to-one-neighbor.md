---
id: 20260809_223805_stage-6-directed-messaging-one-baby-whispers-to-one-neighbor
title: Stage 6 directed messaging — one baby whispers to one neighbor
status: done
tags: world-realm,communication,stage6,messaging
created: 2026-08-09T22:38:05.351683+00:00
---

Stage 6 directed messaging — one baby whispers to one neighbor

Directed inter-agent messaging completes Stage 6 Communication: a baby can address ONE specific neighbor instead of broadcasting to the whole grid. Opt-in so the locked selection proofs keep their exact genome layout.

Design (simulation.py):
- WorldParams: message_enabled (default False), message_cost=0.5 (energy per amplitude unit), message_range=5.0, message_gate_threshold=0.5.
- SimBaby gains perceptron_message (entity-input -> 1 gate), _inbox (sender_id -> amplitude), decide_message(other) returns gate value if it clears the threshold else 0.0. Shared _entity_features(entity) helper feeds social_step, learn, and decide_message; message amplitude is the entity feature at index 5, visible only to brains with entity_input_dim >= 6.
- One-tick latency: SimScene._pending_messages (sender, target, amplitude) posted during step 4c (after movement); deliver_messages() clears every inbox and routes at the start of the next tick. Target = neediest nearby baby within message_range; strongest amplitude wins on duplicate sender->target; dead targets dropped; sender pays message_cost*amplitude.
- Learning: message perceptron delta-rule updated in learn() alongside the entity perceptron. Persistence: SimBaby/SimScene to_dict/from_dict round-trip the message brain (None-safe) and the pending bus.

Genome integration (evolution.py): message tensors appended LAST in from_baby (keyed on the baby's perceptron), random (shapes dict), and apply_to -- RNG draw order of the locked proofs is preserved when messages are off.

Verification:
- Locked benchmarks unchanged (messages off): benchmark_social(seed=1) -> ind_coop=0.0000 grp_coop=0.0312 ind_contest=0.1823 grp_contest=0.0260; emergence seeds 1-5 identical.
- New TestDirectedMessage suite (11 tests): off-by-default guard, gate-gated emission, amplitude = sigmoid(gate), one-tick-latent delivery to neediest neighbor, message feature at index 5, dim-5 slicing, genome tensor round-trip, scene pending-bus round-trip, message-brain learning.
- World-realm suite: 298 passed (test_evolution 57 -> 68).

CLI: world_driver.py --messages. Docs: WORLD_REALM.md Stage 6 mechanism + Directed Communication feature row, Not Built table cleared, communication sections updated.