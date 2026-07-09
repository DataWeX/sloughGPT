import json
from test_support import get_test_client

client = get_test_client()

def test_get_metrics_structure():
    resp = client.get('/system/metrics')
    assert resp.status_code == 200, f"Unexpected {resp.status_code}"
    data = resp.json()
    # expected keys
    expected = {'cpu_percent', 'memory_percent', 'memory_used_gb', 'memory_total_gb'}
    assert expected.issubset(data.keys()), f"Missing keys {expected - data.keys()}"
    # basic type checks
    assert isinstance(data['cpu_percent'], (int, float))
    assert isinstance(data['memory_used_gb'], (int, float))

def test_get_info_structure():
    resp = client.get('/system/info')
    assert resp.status_code == 200
    data = resp.json()
    expected = {'platform', 'platform_release', 'platform_version', 'architecture', 'processor', 'cpu_count'}
    assert expected.issubset(data.keys())
    assert isinstance(data['platform'], str)
    assert isinstance(data['cpu_count'], int)

def test_get_disk_structure():
    resp = client.get('/system/disk')
    assert resp.status_code == 200
    data = resp.json()
    expected = {'total_gb', 'used_gb', 'free_gb', 'percent'}
    assert expected.issubset(data.keys())
    # disk percent should be between 0 and 100
    assert 0 <= data['percent'] <= 100


def test_get_lifecycle_structure():
    """Lifecycle endpoint returns expected fields even without full startup."""
    resp = client.get('/system/lifecycle')
    assert resp.status_code == 200
    data = resp.json()
    assert 'phase' in data
    assert 'profile' in data
    assert 'uptime' in data
    assert 'in_flight' in data
    assert 'hooks' in data
    assert 'gates' in data
    # Phase will be init since lifespan hasn't run
    assert data['phase'] == 'init'
    assert data['profile'] == 'full'
    assert data['in_flight'] == 0
    assert 'startup' in data['hooks']
    assert 'shutdown' in data['hooks']
    assert 'preview' in data['hooks']
    assert 'total' in data['gates']
