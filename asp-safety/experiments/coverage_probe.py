"""Coverage probe (T-2/P-9): catalog finding names vs test assertions."""

import re
import glob

findings = set()
for f in glob.glob("rules/*.lp"):
    src = open(f).read()
    for m in re.finditer(
            r"(?:violation|hazard|warning)\(\s*([a-z][a-z_0-9]+)", src):
        findings.add(m.group(1))
# insufficient() first arg is a hazard name, not a rule predicate.
findings.discard("emergency_brake")

asserted = set()
for f in glob.glob("tests/test_*.py"):
    src = open(f).read()
    for m in re.finditer(r"['\"]([a-z_]+)['\"]", src):
        if m.group(1) in findings:
            asserted.add(m.group(1))

missing = sorted(findings - asserted)
print(f"catalog findings: {len(findings)}, asserted in tests: {len(asserted)}")
print("MISSING firing assertions:")
for m in missing:
    print("  ", m)
