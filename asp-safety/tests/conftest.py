"""Pytest collection tweaks for the ASP safety-layer suite.

Perf-budget tests (test_performance.py) measure wall-clock solve
latency against fixed budgets (50 ms bare / 100 ms 20-agent). Run
them FIRST so they measure on a cold machine: alphabetically the
suite starts with test_all.py, whose torch CPU work (planner build +
E2E run) heats the machine and pushes the timer past the gate even
with best-of-N sampling (read-time-only flake, green standalone).
Ordering is the fix; budgets are not inflated and real headroom is
still P-8/PR-3 work.
"""


def pytest_collection_modifyitems(items):
    perf = [i for i in items if "test_performance" in str(i.fspath)]
    rest = [i for i in items if "test_performance" not in str(i.fspath)]
    items[:] = perf + rest
