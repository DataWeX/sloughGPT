"""Tests for serving profiles module."""

from domains.infrastructure.serving_profiles import (
    ProfileTier,
    ServingProfile,
    apply_profile,
    detect_recommended_profile,
    get_active_profile_id,
    get_profile,
    list_profiles,
)


class TestServingProfile:
    def test_to_dict_serializes_tier(self):
        p = get_profile("balanced")
        assert p is not None
        d = p.to_dict()
        assert d["tier"] == "balanced"
        assert isinstance(d["tags"], list)

    def test_to_dict_has_all_fields(self):
        p = get_profile("cpu_only")
        assert p is not None
        d = p.to_dict()
        required = [
            "id", "name", "description", "tier", "device", "quantize",
            "quant_bits", "inference_pool_size", "enable_guard", "tags",
        ]
        for key in required:
            assert key in d, f"missing key: {key}"


class TestListProfiles:
    def test_returns_all_built_in_profiles(self):
        profiles = list_profiles()
        assert len(profiles) >= 5
        ids = {p["id"] for p in profiles}
        assert "cpu_only" in ids
        assert "cpu_optimized" in ids
        assert "balanced" in ids
        assert "gpu_performance" in ids
        assert "training" in ids

    def test_returns_dicts(self):
        profiles = list_profiles()
        for p in profiles:
            assert isinstance(p, dict)
            assert "id" in p
            assert "name" in p


class TestGetProfile:
    def test_returns_profile(self):
        p = get_profile("balanced")
        assert p is not None
        assert p.id == "balanced"
        assert p.name == "Balanced"

    def test_returns_none_for_unknown(self):
        assert get_profile("nonexistent") is None

    def test_all_profiles_have_valid_tier(self):
        for p_dict in list_profiles():
            p = get_profile(p_dict["id"])
            assert p is not None
            assert isinstance(p.tier, ProfileTier)


class TestApplyProfile:
    def test_apply_balanced(self):
        result = apply_profile("balanced")
        assert result["active_profile_id"] == "balanced"
        assert result["profile"]["id"] == "balanced"
        assert "live_settings" in result
        assert "requires_restart" in result
        assert isinstance(result["live_settings"], dict)
        assert isinstance(result["requires_restart"], list)

    def test_apply_sets_active(self):
        apply_profile("cpu_only")
        assert get_active_profile_id() == "cpu_only"
        apply_profile("balanced")
        assert get_active_profile_id() == "balanced"

    def test_apply_unknown_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown profile"):
            apply_profile("nonexistent")

    def test_apply_live_settings_include_generation(self):
        result = apply_profile("gpu_performance")
        live = result["live_settings"]
        assert "temperature" in live
        assert "top_p" in live
        assert "top_k" in live
        assert "max_new_tokens" in live

    def test_apply_live_settings_include_guard(self):
        result = apply_profile("cpu_only")
        live = result["live_settings"]
        assert "enable_guard" in live
        assert live["enable_guard"] is False  # cpu_only has guard disabled

    def test_apply_live_settings_include_memory_pressure(self):
        result = apply_profile("gpu_performance")
        live = result["live_settings"]
        assert "memory_pressure_warning" in live
        assert "memory_pressure_critical" in live
        assert "memory_pressure_emergency" in live

    def test_restart_settings_listed(self):
        result = apply_profile("training")
        restart = result["requires_restart"]
        assert "device" in restart
        assert "compute_threads" in restart
        assert "workload_mode" in restart


class TestDetectRecommendedProfile:
    def test_gpu_high_ram(self):
        assert detect_recommended_profile(16, has_gpu=True) == "gpu_performance"

    def test_no_gpu_high_ram(self):
        assert detect_recommended_profile(16, has_gpu=False) == "balanced"

    def test_medium_ram(self):
        assert detect_recommended_profile(8, has_gpu=False) == "cpu_optimized"

    def test_low_ram(self):
        assert detect_recommended_profile(4, has_gpu=False) == "cpu_only"

    def test_gpu_low_ram_still_cpu(self):
        assert detect_recommended_profile(4, has_gpu=True) == "cpu_only"
