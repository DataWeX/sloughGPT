from domains.infrastructure.correlation import set_correlation_id, get_correlation_id


class TestCorrelation:
    def test_default_is_none(self):
        assert get_correlation_id() is None

    def test_set_and_get(self):
        set_correlation_id("req-abc-123")
        assert get_correlation_id() == "req-abc-123"

    def test_set_none(self):
        set_correlation_id("first")
        set_correlation_id(None)
        assert get_correlation_id() is None

    def test_overwrite(self):
        set_correlation_id("a")
        set_correlation_id("b")
        assert get_correlation_id() == "b"

    def test_empty_string(self):
        set_correlation_id("")
        assert get_correlation_id() == ""
