"""
End-to-end DPO trainer verification on the migrated NVMe root.

Builds a real SloLSTM + a temp sqlite FeedbackDB with 2 thumbs_up and 2
thumbs_down records, then runs HFDPOTrainer.train() and asserts:
  - status == "accepted"
  - pairs are built from the store
  - chosen log-prob increases and rejected log-prob decreases (preference learned)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "packages", "core-py"))

import numpy as np

import domains.feedback.database as database
from domains.feedback.hf_dpo import HFDPOTrainer
from domains.training.slonet import SloLSTM


class CharTokenizer:
    def encode(self, text):
        return [ord(ch) for ch in text if ord(ch) < 128]


def main():
    tmp = tempfile.mkdtemp(prefix="dpo_verify_")
    database._feedback_db = None
    db = database.get_feedback_db(db_path=os.path.join(tmp, "feedback.db"))

    c1 = db.create_conversation(user_id="test", title="conv1")
    c2 = db.create_conversation(user_id="test", title="conv2")

    m_up1 = db.add_message(c1, "assistant", "Helpful detail on gradient flow")
    m_up2 = db.add_message(c2, "assistant", "A clear and complete answer")
    m_dn1 = db.add_message(c1, "assistant", "Too vague, did not answer")
    m_dn2 = db.add_message(c1, "assistant", "Confusing and wrong math")

    db.add_feedback(m_up1, "thumbs_up")
    db.add_feedback(m_up2, "thumbs_up")
    db.add_feedback(m_dn1, "thumbs_down")
    db.add_feedback(m_dn2, "thumbs_down")

    model = SloLSTM(vocab_size=256, embed_dim=32, hidden_dim=64, num_layers=1, dropout=0.0)
    trainer = HFDPOTrainer(model=model, tokenizer=CharTokenizer())

    pairs = trainer.prepare_dpo_pairs()
    print("pairs built:", len(pairs))
    for p in pairs:
        print("  chosen:", p["chosen"], "| rejected:", p["rejected"])

    chosen_ids = trainer._encode(pairs[0]["chosen"])
    rejected_ids = trainer._encode(pairs[0]["rejected"])
    before = (trainer._log_probs(chosen_ids), trainer._log_probs(rejected_ids))
    print("logp before:", before)

    result = trainer.train(pairs=pairs)
    print("train result:", result)

    after = (trainer._log_probs(chosen_ids), trainer._log_probs(rejected_ids))
    print("logp after:", after)

    assert result["status"] == "accepted", f"expected accepted, got {result}"
    assert result["pairs_trained"] >= 2
    assert result["avg_loss"] is not None and result["avg_loss"] >= 0
    assert after[0] > before[0], f"chosen logp did not increase: {before[0]} -> {after[0]}"
    assert after[1] < before[1], f"rejected logp did not decrease: {before[1]} -> {after[1]}"
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
