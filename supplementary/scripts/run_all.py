# -*- coding: utf-8 -*-
"""Run every supplement check and report a single PASS/FAIL summary."""
import subprocess, sys, os
here = os.path.dirname(os.path.abspath(__file__))
order = ["check_algebra.py", "check_spectral.py", "check_gelfand.py",
         "check_chi2.py", "check_certificates.py"]
bad = []
for f in order:
    print("\n" + "=" * 66)
    print("RUN", f)
    print("=" * 66)
    r = subprocess.run([sys.executable, os.path.join(here, f)], cwd=here)
    if r.returncode != 0:
        bad.append(f)
print("\n" + "=" * 66)
print("SUPPLEMENT VERIFICATION:", "ALL PASS" if not bad else "FAILED: " + ", ".join(bad))
print("=" * 66)
sys.exit(1 if bad else 0)
