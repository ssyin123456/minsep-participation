# Numerical cross-checks

Shared core: `scripts/common.py`

| function | purpose |
|---|---|
| `lam_star(N,R)` | largest root modulus of the degree-(R+1) companion polynomial (`mpmath` for R>40) |
| `tau_int(N,R)`, `sigma2`, `cov0` | closed forms of the manuscript |
| `delta_seq`, `prop4_ratio`, `prop4_ratio_by_series` | Proposition 4 and an independent truncated-covariance verification |
| `direct_tv(N,R)` | full sparse `P_R` iteration (small N) |
| `correct_lump_tv(N,R)` | identity-tracking lump, `D_R = sum_k C(R,k)(R)_k` states, independent of N |

Checks performed before the experiments were run:

- `correct_lump_tv` and `direct_tv` agree to 5.5e-16 on all Table III chains
  with R<=7 and reproduce every exact `tau_mix` reported there;
  `(10,8)=63` is checked by `direct_tv` alone.
- `prop4_ratio` and `prop4_ratio_by_series` agree to 3e-13 over the whole E3
  grid; both return `0.168944262` at `(N,R,eta)=(20,16,0.04)`.
- The R=1 aggregate criterion agrees with the exact N-state Poisson solve to
  5.6e-16 over 36 profiles.

An identity-only "bit-mask" lump (recording special-vs-fresh but not *which*
`a_i`) underestimates the total-variation distance and is not used. Exact
`tau_mix` at N=100 is reported only for R<=7; for larger R both `(N)_R` and
`D_R` are too large, so Fig. 6(b) shows only the provable lower bound and the
leading term `R+L`. The centered observable must be
`f_i(w) = 1{w=i} - pi_i` with a constant `pi_i`, not `pi_w`.

The closed forms above are checked by `supplementary/scripts/check_*.py`
(Appendix C) and by `scripts/common.py`.
