import numpy as np
import pytest

from domains.collections.baby_social import (
    SocialConfig, Message, BabySocial, BabyCultural, BabySocialSystem,
)


class FakeBaby:
    def __init__(self, id=1, position=None, energy=50.0):
        self.entity = type('Entity', (), {'id': id, 'entity_type': 1})()
        self.position = np.array(position if position else [32, 0, 32], dtype=np.float64)
        self.energy = energy
        self.alive = True


class TestMessage:
    def test_create_message(self):
        msg = Message(sender_id=1, receiver_id=2, content={"text": "hello"}, timestamp=0.0)
        assert msg.sender_id == 1
        assert msg.receiver_id == 2
        assert msg.content["text"] == "hello"


class TestBabySocial:
    def test_send_message(self):
        baby = FakeBaby(id=1)
        social = BabySocial(baby)
        msg = social.send_message(2, {"text": "hello"})
        assert msg.sender_id == 1
        assert msg.receiver_id == 2
        assert len(social._outbox) == 1

    def test_receive_message(self):
        baby = FakeBaby(id=1)
        social = BabySocial(baby)
        msg = Message(sender_id=2, receiver_id=1, content={"text": "hello"}, timestamp=0.0)
        social.receive_message(msg)
        assert len(social._inbox) == 1

    def test_get_messages(self):
        baby = FakeBaby(id=1)
        social = BabySocial(baby)
        msg = Message(sender_id=2, receiver_id=1, content={"text": "hello"}, timestamp=0.0)
        social.receive_message(msg)
        messages = social.get_messages()
        assert len(messages) == 1
        assert len(social._inbox) == 0

    def test_trust(self):
        baby = FakeBaby(id=1)
        social = BabySocial(baby)
        assert social.get_trust(2) == 0.5
        social.update_trust(2, 0.2)
        assert abs(social.get_trust(2) - 0.7) < 1e-6
        social.update_trust(2, -0.5)
        assert abs(social.get_trust(2) - 0.2) < 1e-6

    def test_trust_bounds(self):
        baby = FakeBaby(id=1)
        social = BabySocial(baby)
        social.update_trust(2, 1.0)
        assert social.get_trust(2) == 1.0
        social.update_trust(2, -2.0)
        assert social.get_trust(2) == 0.0

    def test_observe_action(self):
        baby = FakeBaby(id=1)
        other = FakeBaby(id=2)
        social = BabySocial(baby)
        social.observe_action(other, "move", {"success": True})
        assert len(social._social_memory) == 1

    def test_try_imitate_out_of_range(self):
        baby = FakeBaby(id=1, position=[0, 0, 0])
        other = FakeBaby(id=2, position=[100, 0, 100])
        config = SocialConfig(imitation_chance=1.0)
        social = BabySocial(baby, config)
        result = social.try_imitate(other, None)
        assert not result["success"]
        assert result["reason"] == "out_of_range"

    def test_try_imitate_low_trust(self):
        baby = FakeBaby(id=1)
        other = FakeBaby(id=2, position=[33, 0, 32])
        config = SocialConfig(imitation_chance=1.0)
        social = BabySocial(baby, config)
        social.update_trust(2, -0.5)
        result = social.try_imitate(other, None)
        assert not result["success"]
        assert result["reason"] == "low_trust"

    def test_teach_out_of_range(self):
        baby = FakeBaby(id=1, position=[0, 0, 0])
        other = FakeBaby(id=2, position=[100, 0, 100])
        social = BabySocial(baby)
        result = social.teach(other, {"lesson": "hello"})
        assert not result["success"]
        assert result["reason"] == "out_of_range"

    def test_teach_insufficient_energy(self):
        baby = FakeBaby(id=1, energy=1.0)
        other = FakeBaby(id=2, position=[33, 0, 32])
        social = BabySocial(baby)
        result = social.teach(other, {"lesson": "hello"})
        assert not result["success"]
        assert result["reason"] == "insufficient_energy"

    def test_summary(self):
        baby = FakeBaby(id=1)
        social = BabySocial(baby)
        s = social.summary()
        assert "inbox_size" in s
        assert "teach_count" in s
        assert "learn_count" in s


class TestBabyCultural:
    def test_record_culture(self):
        cultural = BabyCultural()
        baby = FakeBaby()
        cultural.record_culture(baby, {"total": 5}, {"preferred": 1})
        assert len(cultural._cultural_memory) == 1

    def test_end_generation(self):
        cultural = BabyCultural()
        baby = FakeBaby()
        cultural.record_culture(baby, {}, {})
        cultural.end_generation()
        assert len(cultural._generations) == 1
        assert len(cultural._current_generation) == 0

    def test_generation_stats(self):
        cultural = BabyCultural()
        baby1 = FakeBaby(id=1, energy=100.0)
        baby2 = FakeBaby(id=2, energy=50.0)
        cultural.record_culture(baby1, {}, {})
        cultural.record_culture(baby2, {}, {})
        cultural.end_generation()
        stats = cultural.get_generation_stats()
        assert stats["size"] == 2
        assert stats["avg_energy"] == 75.0

    def test_cultural_trend(self):
        cultural = BabyCultural()
        baby = FakeBaby()
        cultural.record_culture(baby, {}, {})
        cultural.end_generation()
        trend = cultural.get_cultural_trend()
        assert trend["generations"] == 1

    def test_top_performers(self):
        cultural = BabyCultural()
        baby1 = FakeBaby(id=1, energy=100.0)
        baby2 = FakeBaby(id=2, energy=50.0)
        baby3 = FakeBaby(id=3, energy=75.0)
        cultural.record_culture(baby1, {}, {})
        cultural.record_culture(baby2, {}, {})
        cultural.record_culture(baby3, {}, {})
        top = cultural.get_top_performers(2)
        assert len(top) == 2
        assert top[0]["energy"] == 100.0

    def test_summary(self):
        cultural = BabyCultural()
        baby = FakeBaby()
        cultural.record_culture(baby, {}, {})
        s = cultural.summary()
        assert s["total_records"] == 1


class TestBabySocialSystem:
    def test_register_baby(self):
        system = BabySocialSystem()
        baby = FakeBaby(id=1)
        system.register_baby(baby)
        assert 1 in system._social_modules
        assert hasattr(baby, '_social')

    def test_unregister_baby(self):
        system = BabySocialSystem()
        baby = FakeBaby(id=1)
        system.register_baby(baby)
        system.unregister_baby(baby)
        assert 1 not in system._social_modules

    def test_get_social(self):
        system = BabySocialSystem()
        baby = FakeBaby(id=1)
        system.register_baby(baby)
        social = system.get_social(baby)
        assert social is not None

    def test_tick(self):
        system = BabySocialSystem()
        baby1 = FakeBaby(id=1, position=[32, 0, 32])
        baby2 = FakeBaby(id=2, position=[33, 0, 32])
        system.register_baby(baby1)
        system.register_baby(baby2)
        results = system.tick([baby1, baby2], None)
        assert len(results) == 2

    def test_tick_with_messages(self):
        system = BabySocialSystem()
        baby1 = FakeBaby(id=1, position=[32, 0, 32])
        baby2 = FakeBaby(id=2, position=[33, 0, 32])
        system.register_baby(baby1)
        system.register_baby(baby2)
        social1 = system.get_social(baby1)
        social1.send_message(2, {"text": "hello"})
        msg = social1._outbox[0]
        social2 = system.get_social(baby2)
        social2.receive_message(msg)
        results = system.tick([baby1, baby2], None)
        assert len(results) == 2

    def test_end_generation(self):
        system = BabySocialSystem()
        baby = FakeBaby(id=1)
        system.register_baby(baby)
        system.record_culture([baby])
        system.end_generation()
        assert len(system._cultural._generations) == 1

    def test_record_culture(self):
        system = BabySocialSystem()
        baby = FakeBaby(id=1)
        system.register_baby(baby)
        system.record_culture([baby])
        assert len(system._cultural._cultural_memory) == 1

    def test_summary(self):
        system = BabySocialSystem()
        baby = FakeBaby(id=1)
        system.register_baby(baby)
        s = system.summary()
        assert s["social_modules"] == 1
        assert "cultural" in s
