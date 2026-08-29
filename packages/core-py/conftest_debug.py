import time
import sys
import os

_log = open('/tmp/test_execution.log', 'w')


def pytest_runtest_setup(item):
    _log.write(f'SETUP {item.nodeid}\n')
    _log.flush()


def pytest_runtest_call(item):
    _log.write(f'CALL {item.nodeid}\n')
    _log.flush()


def pytest_runtest_teardown(item):
    _log.write(f'TEARDOWN {item.nodeid}\n')
    _log.flush()


def pytest_sessionfinish(session, exitstatus):
    _log.close()
