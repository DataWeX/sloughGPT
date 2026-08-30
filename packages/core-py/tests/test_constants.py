"""Tests for infrastructure constants — validates all shared constants exist and have sane values."""

from domains.infrastructure.constants import (
    DEFAULT_GENERATE_TIMEOUT,
    DEFAULT_STALL_TIMEOUT,
    DEFAULT_STARTUP_TIMEOUT,
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_LOAD_MAX_RETRIES,
    DEFAULT_LOAD_RETRY_DELAY,
)


class TestGenerateTimeout:
    def test_positive(self):
        assert DEFAULT_GENERATE_TIMEOUT > 0

    def test_is_float(self):
        assert isinstance(DEFAULT_GENERATE_TIMEOUT, float)

    def test_reasonable_range(self):
        assert 1.0 <= DEFAULT_GENERATE_TIMEOUT <= 600.0

    def test_exact_value(self):
        assert DEFAULT_GENERATE_TIMEOUT == 120.0

    def test_not_zero(self):
        assert DEFAULT_GENERATE_TIMEOUT != 0.0

    def test_not_negative(self):
        assert DEFAULT_GENERATE_TIMEOUT >= 0.0


class TestStallTimeout:
    def test_positive(self):
        assert DEFAULT_STALL_TIMEOUT > 0

    def test_is_float(self):
        assert isinstance(DEFAULT_STALL_TIMEOUT, float)

    def test_exact_value(self):
        assert DEFAULT_STALL_TIMEOUT == 120.0

    def test_not_zero(self):
        assert DEFAULT_STALL_TIMEOUT != 0.0

    def test_reasonable_range(self):
        assert 1.0 <= DEFAULT_STALL_TIMEOUT <= 600.0


class TestStartupTimeout:
    def test_positive(self):
        assert DEFAULT_STARTUP_TIMEOUT > 0

    def test_is_float(self):
        assert isinstance(DEFAULT_STARTUP_TIMEOUT, float)

    def test_ge_generate(self):
        assert DEFAULT_STARTUP_TIMEOUT >= DEFAULT_GENERATE_TIMEOUT

    def test_exact_value(self):
        assert DEFAULT_STARTUP_TIMEOUT == 300.0

    def test_reasonable_range(self):
        assert 1.0 <= DEFAULT_STARTUP_TIMEOUT <= 3600.0


class TestIdleTimeout:
    def test_positive(self):
        assert DEFAULT_IDLE_TIMEOUT > 0

    def test_is_float(self):
        assert isinstance(DEFAULT_IDLE_TIMEOUT, float)

    def test_exact_value(self):
        assert DEFAULT_IDLE_TIMEOUT == 300.0

    def test_reasonable_range(self):
        assert 1.0 <= DEFAULT_IDLE_TIMEOUT <= 3600.0

    def test_ge_generate(self):
        assert DEFAULT_IDLE_TIMEOUT >= DEFAULT_GENERATE_TIMEOUT


class TestLoadMaxRetries:
    def test_nonneg(self):
        assert DEFAULT_LOAD_MAX_RETRIES >= 0

    def test_is_int(self):
        assert isinstance(DEFAULT_LOAD_MAX_RETRIES, int)

    def test_exact_value(self):
        assert DEFAULT_LOAD_MAX_RETRIES == 2

    def test_reasonable_range(self):
        assert 0 <= DEFAULT_LOAD_MAX_RETRIES <= 10


class TestLoadRetryDelay:
    def test_positive(self):
        assert DEFAULT_LOAD_RETRY_DELAY > 0

    def test_is_float(self):
        assert isinstance(DEFAULT_LOAD_RETRY_DELAY, float)

    def test_exact_value(self):
        assert DEFAULT_LOAD_RETRY_DELAY == 5.0

    def test_reasonable_range(self):
        assert 0.1 <= DEFAULT_LOAD_RETRY_DELAY <= 60.0


class TestConstantsCrossChecks:
    def test_startup_ge_stall(self):
        assert DEFAULT_STARTUP_TIMEOUT >= DEFAULT_STALL_TIMEOUT

    def test_idle_ge_generate(self):
        assert DEFAULT_IDLE_TIMEOUT >= DEFAULT_GENERATE_TIMEOUT

    def test_retry_delay_less_than_generate(self):
        assert DEFAULT_LOAD_RETRY_DELAY < DEFAULT_GENERATE_TIMEOUT

    def test_generate_and_stall_equal(self):
        assert DEFAULT_GENERATE_TIMEOUT == DEFAULT_STALL_TIMEOUT

    def test_startup_equal_idle(self):
        assert DEFAULT_STARTUP_TIMEOUT == DEFAULT_IDLE_TIMEOUT

    def test_all_are_numeric(self):
        assert isinstance(DEFAULT_GENERATE_TIMEOUT, (int, float))
        assert isinstance(DEFAULT_STALL_TIMEOUT, (int, float))
        assert isinstance(DEFAULT_STARTUP_TIMEOUT, (int, float))
        assert isinstance(DEFAULT_IDLE_TIMEOUT, (int, float))
        assert isinstance(DEFAULT_LOAD_MAX_RETRIES, (int, float))
        assert isinstance(DEFAULT_LOAD_RETRY_DELAY, (int, float))

    def test_no_none_values(self):
        assert DEFAULT_GENERATE_TIMEOUT is not None
        assert DEFAULT_STALL_TIMEOUT is not None
        assert DEFAULT_STARTUP_TIMEOUT is not None
        assert DEFAULT_IDLE_TIMEOUT is not None
        assert DEFAULT_LOAD_MAX_RETRIES is not None
        assert DEFAULT_LOAD_RETRY_DELAY is not None

    def test_total_retry_time_reasonable(self):
        total = DEFAULT_LOAD_RETRY_DELAY * (2 ** DEFAULT_LOAD_MAX_RETRIES)
        assert total < DEFAULT_GENERATE_TIMEOUT

    def test_import_all_names(self):
        import domains.infrastructure.constants as c
        assert hasattr(c, "DEFAULT_GENERATE_TIMEOUT")
        assert hasattr(c, "DEFAULT_STALL_TIMEOUT")
        assert hasattr(c, "DEFAULT_STARTUP_TIMEOUT")
        assert hasattr(c, "DEFAULT_IDLE_TIMEOUT")
        assert hasattr(c, "DEFAULT_LOAD_MAX_RETRIES")
        assert hasattr(c, "DEFAULT_LOAD_RETRY_DELAY")

    def test_module_docstring(self):
        import domains.infrastructure.constants as c
        assert c.__doc__ is not None
        assert len(c.__doc__) > 0
