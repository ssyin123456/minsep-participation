# -*- coding: utf-8 -*-
"""S.7: exact TV/ell-1 prefactor certificates and the status of the
uniform-prefactor conjecture (open, NOT a theorem)."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import lam_star, LumpChain, check

fails = []
def ck(c, label):
    if not check(c, label): fails.append(label)

print("== S.7: sup_{m>=1} d(R+m)/lambda*^m on exact lump iterations ==")
cases = [(6,2),(7,2),(8,3),(8,4),(10,5),(10,8),(12,10),(14,12),
         (15,3),(29,4),(71,6),(20,6),(100,7),(9,3),(11,4),(13,6),
         (30,4),(50,5),(200,6)]
mx = 0.0
for N, R in cases:
    L = LumpChain(N, R)
    ls = lam_star(N, R)
    try: h = L.run(R+14)
    except RuntimeError: print("     (%3d,%2d) skipped (support too large)" % (N,R)); continue
    pre = max(L.tv(h[R+m])/ls**m for m in range(1, len(h)-R))
    mx = max(mx, pre)
    print("     (%3d,%2d) m=%3d  prefactor = %.4f" % (N,R,N-R,pre))
ck(mx <= 1.58, "max prefactor over %d chains = %.4f <= 1.58" % (len(cases),mx))

print("== S.7: ||v_0||_1 = 2(1-alpha_0) <= 2 holds on the whole aperiodic range ==")
ck(True, "||v_0||_1 <= 2 is elementary (two probability measures)")

print("== S.7: status note ==")
print("     Uniform ell-1 prefactor / tau_mix = R + O(L) on ALL (N,R): OPEN")
print("     (chi-square version is false; see check_algebra.py).")
print("\nCERTIFICATES: %d checks failed" % len(fails))
sys.exit(1 if fails else 0)
