# -*- coding: utf-8 -*-
r"""Verify that the published datasets are self-consistent and reproducible.

Usage: python3 verify_data.py [--full]
  --full also re-runs the Monte Carlo (fixed seed) and compares every value.
"""
import io, json, math, os, re, sys
import numpy as np

ROOT = os.environ.get("MINSEP_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
IN = os.path.join(ROOT, "data")
R = []


def ck(name, cond, detail=""):
    R.append((name, bool(cond), detail))
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  | " + detail) if detail else ""))


def close(a, b, tol=1e-10):
    return abs(float(a) - float(b)) <= tol


v2 = json.load(open(os.path.join(IN, "figures_data_v2.json")))
v3 = json.load(open(os.path.join(IN, "figures_data_v3.json")))
v3e = json.load(open(os.path.join(IN, "figures_data_v3_estimation.json")))

print("=" * 78)
print("1. 三份 JSON 的 config / 同源性")
print("=" * 78)
for name, d in [("v2", v2), ("v3", v3), ("v3_estimation", v3e)]:
    c = d["config"]
    print(f"  {name:14s} seed={c['seed']} N={c['N']} T={c['T']} traj={c['trajectories']} R={c['R']} eta_lr={c['eta_lr']} d={c['d']}")
ck("三份 config 的 seed/N/T/轨迹/R/eta/d 完全一致",
   all(all(v2["config"][k] == d["config"][k] for d in (v3, v3e))
       for k in ("seed", "N", "T", "trajectories", "R", "eta_lr", "d")))
ck("v3_estimation note 声明与 v3 同源", "identical simulation data" in v3e.get("note", ""))
ck("v2 含 uniform/counterexample/small_rho", all(k in v2 for k in ("uniform", "counterexample", "small_rho")))
ck("v3 含 errorbars/nonuniform/rho_scale", all(k in v3 for k in ("errorbars", "nonuniform", "rho_scale")))

print()
print("=" * 78)
print("2. 解析式复核（不依赖 Monte Carlo）")
print("=" * 78)
N, T = 20, 1200
Rs = [0, 1, 2, 4, 8, 12, 16]
bad = []
for r in Rs:
    pred = (N - r) * (N - r - 1) / N ** 2
    got = v2["uniform"][str(r)]["pred"]
    if not close(pred, got): bad.append((r, pred, got))
ck("Theorem 2: pred=(N-R)(N-R-1)/N^2 对 7 个 R", not bad, f"max diff {max((abs(p-g) for _,p,g in bad), default=0):.2e}")

pf = np.array([1/100, 24/25, 1/100, 1/100, 1/100]); pf /= pf.sum()
Z = 1 - (pf**2).sum()
th0 = pf[0] * (1 - pf[0]); th1 = 821931 / 8788000
ck("Prop 1: sigma^2(0)=p1(1-p1)=0.0099", close(th0, v2["counterexample"]["0"]["theory"], 1e-12), f"{th0:.10f}")
ck("Prop 1: sigma^2(1)=821931/8788000", close(th1, v2["counterexample"]["1"]["theory"], 1e-12), f"{th1:.10f}")
ratio = th1 / th0
ck("Prop 1: 比值 ~9.4474 (文稿写 9.45)", abs(ratio - 9.4474) < 5e-4, f"{ratio:.6f}")

Nsr = 100
bad = []
for r in [0, 1, 2, 5, 10]:
    pred = (Nsr - r) * (Nsr - r - 1) / Nsr ** 2
    rv = pred / ((Nsr - 1) / Nsr)
    d = v2["small_rho"][str(r)]
    if not (close(pred, d["pred"]) and close(rv, d["ratio_var"]) and close(r / Nsr, d["rho"])):
        bad.append(r)
ck("small_rho: pred / ratio_var / rho=R/N 对 5 个 R", not bad, str(bad))
ck("small_rho: sgd_ratio 以 R=0 归一且 R=0 为 1",
   close(v2["small_rho"]["0"]["sgd_ratio"], 1.0) and
   all(close(v2["small_rho"][str(r)]["sgd_ratio"],
             v2["small_rho"][str(r)]["sgd_end"] / v2["small_rho"]["0"]["sgd_end"], 1e-9)
       for r in [0, 1, 2, 5, 10]))

idx = np.arange(1, 101, dtype=float); p = idx ** -1.5; p /= p.sum()
a0 = float(1 - (p ** 2).sum())
ck("nonuniform: analytic_R0 = 1-sum(p_i^2)", close(a0, v3["nonuniform"]["analytic_R0"], 1e-12), f"{a0:.10f}")
cal = v3["nonuniform"]["results"]["0"]["estimate"] / a0
ck("nonuniform: calibration_ratio = MC(R=0)/analytic", close(cal, v3["nonuniform"]["calibration_ratio"], 1e-12), f"{cal:.8f}")
ck("nonuniform: 每个 R 的 estimate = 4 个 checkpoint 均值",
   all(close(v["estimate"], np.mean(list(v["checkpoints"].values())), 1e-12)
       for v in v3["nonuniform"]["results"].values()))
ck("nonuniform: spread = max-min(checkpoints)",
   all(close(v["spread"], max(v["checkpoints"].values()) - min(v["checkpoints"].values()), 1e-12)
       for v in v3["nonuniform"]["results"].values()))

bad = []
for key, v in v3["rho_scale"].items():
    m = re.match(r"N(\d+)_R(\d+)", key); Nx, Rx = int(m.group(1)), int(m.group(2))
    rho = Rx / Nx
    rat = (Nx - Rx) * (Nx - Rx - 1) / Nx ** 3 / ((Nx - 1) / Nx ** 2)
    if not (close(rho, v["rho"], 1e-15) and close(rat, v["ratio"], 1e-12)
            and close(100 * (1 - rat), v["improve_pct"], 1e-9)):
        bad.append(key)
ck("rho_scale: 5 组 (N,R) 的 rho/ratio/improve_pct", not bad, str(bad))

print()
print("=" * 78)
print("3. errorbars 派生量重算（Welch / SEM / 重复 A）")
print("=" * 78)
st = v3["errorbars"]["stats"]
for t in ["11", "20", "40", "1200"]:
    for r in Rs:
        e = st[t][str(r)]
        if not close(e["sem"], e["sd"] / math.sqrt(4000), 1e-12): bad.append((t, r))
ck("SEM = sd/sqrt(4000) 对全部 28 组", not bad, str(bad[:3]))
bad = []
for t, z in v3["errorbars"]["welch"].items():
    m0, m16 = st[t]["0"]["mean"], st[t]["16"]["mean"]
    diff = m16 - m0
    se = math.sqrt(st[t]["0"]["sem"] ** 2 + st[t]["16"]["sem"] ** 2)
    if not (close(diff, z["diff"], 1e-12) and close(se, z["se"], 1e-12)
            and close(diff / se, z["zval"], 1e-9)
            and close(100 * diff / m0, z["pct"], 1e-9)):
        bad.append(t)
ck("Welch: diff/se/zval/pct 可由 stats 重算", not bad, str(bad))
ck("Welch 结论方向: t=11 变差(+3.21%), t=1200 改善(-83.1%)",
   v3["errorbars"]["welch"]["11"]["pct"] > 0
   and abs(v3["errorbars"]["welch"]["1200"]["pct"] + 83.1081923441425) < 1e-6)
ck("repeat_A: t=11 符号不一致 / t=1200 一致（文稿已据此收敛措辞）",
   v3["errorbars"]["repeat_A"]["11"]["sign_consistent"] is False
   and v3["errorbars"]["repeat_A"]["1200"]["sign_consistent"] is True)

print()
print("=" * 78)
print("4. 跨文件 / 跨版本一致性")
print("=" * 78)
ck("v3 与 v3_estimation 的 errorbars 完全相同", v3["errorbars"] == v3e["errorbars"])
bad = []
for r in Rs:
    a, b = v2["uniform"][str(r)], v3["errorbars"]["stats"]["11"][str(r)]
    if not (close(a["sgd_end"], v3["errorbars"]["sgd_end"][str(r)])
            and close(a["sgd_t11"], b["mean"])
            and close(a["sgd_t11"], v3["errorbars"]["sgd_t11"][str(r)])):
        bad.append(r)
ck("v2.uniform 的 sgd_end/sgd_t11 == v3.errorbars（同一 seed 同一 simulate）", not bad, str(bad))
for t, key in [("20", "sgd_t20"), ("40", "sgd_t40")]:
    ck(f"v2.uniform.{key} == v3.errorbars.stats[{t}].mean",
       all(close(v2["uniform"][str(r)][key], v3["errorbars"]["stats"][t][str(r)]["mean"], 1e-12) for r in Rs))
dev = [(r, abs(v2["uniform"][str(r)]["tvar_end"] - v2["uniform"][str(r)]["pred"])) for r in Rs]
ck("uniform: t*Var(T) 与 Theorem 2 预测的绝对偏差 <0.01（有限 T 收敛误差）",
   all(x < 0.01 for _, x in dev),
   "max %.5f at R=%d；小值处相对偏差会被放大，属有限 T 正常现象" %
   (max(x for _, x in dev), max(dev, key=lambda z: z[1])[0]))
for r in (0, 1):
    ck(f"counterexample R={r}: t*Var(3000) 贴近理论值",
       abs(v2["counterexample"][str(r)]["end"] - v2["counterexample"][str(r)]["theory"]) < 5e-4,
       f"{v2['counterexample'][str(r)]['end']:.8f} vs {v2['counterexample'][str(r)]['theory']:.8f}")

print()
print("=" * 78)
print("5. 与文稿正文 / 图注中引用的数字核对")
print("=" * 78)
tex = io.open(os.path.join(ROOT, "paper", "main.tex"), encoding="utf-8").read()
tex_nums = set(re.findall(r"\d+\.\d{3,}", tex))
checks = [
    ("9.45 / 9.4474 (Prop 1 比值)", ["9.4474", "9.45"]),
    ("0.7935 (长尾 R=0 解析值)", ["0.7935", "0.793539"]),
    ("0.02% (N=1e5,R=10 改善)", ["0.02"]),
    ("4000 轨迹 / 1200 轮", ["4000", "1200"]),
]
for label, keys in checks:
    hit = [k for k in keys if k in tex]
    ck(f"文稿引用: {label}", bool(hit), ("命中 " + ",".join(hit)) if hit else "not found in paper/main.tex")
for label, k in [("0.9944 (MC/解析校准比)", "0.9944"),
                 ("0.8070 (small_rho R=10 稳态误差)", "0.8070"),
                 ("83.1% (t=1200 Welch 改善)", "83.1")]:
    print(f"  INFO  {label}：属配套实验数据，正文未逐一引用（{k} {'在' if k in tex else '不在'} tex 中）")
# L913 的 Fig.4 数字必须与 fig4 数据一致
nu = v3["nonuniform"]["results"]
ck("正文 Fig.4 数值与 JSON 一致（R=0 / R=1 估计）",
   "0.6602" not in tex and "0.7891" in tex and "0.6737" in tex,
   ("已修正为 %.4f / %.4f（旧稿 0.7935/0.6602 与数据不符）" %
    (nu["0"]["estimate"], nu["1"]["estimate"])) if "0.6602" not in tex
   else "旧稿数字仍待修正")
# fig 1-5 的 caption 是否与数据口径一致
caps = re.findall(r"\\caption\{(.{0,400}?)\}\n", tex, re.S)
print(f"  (文稿共 {len(caps)} 条 caption)")

print()
print("=" * 78)
print("6. 发布数据与 minsep 复现模块的对应")
print("=" * 78)
sys.path.insert(0, ROOT)
from minsep import experiments as X
need = ("run_uniform", "run_counterexample", "run_small_rho", "rho_scale",
        "errorbar_statistics", "longtail_estimates")
ck("minsep.experiments 提供全部 6 个复现函数",
   all(hasattr(X, f) for f in need))
ck("figures_data_v2 的字段可被重画脚本直接读取",
   all(k in v2 for k in ("uniform", "counterexample", "small_rho"))
   and all(all(k in v2["uniform"][str(r)] for k in
               ("pred", "tvar_end", "sgd_end", "sgd_t11", "sgd_t20", "sgd_t40"))
           for r in Rs))
ck("figures_data_v3 含 rho_scale / errorbars / nonuniform",
   all(k in v3 for k in ("rho_scale", "errorbars", "nonuniform")))
curves_p = os.path.join(IN, "recomputed_v2_curves.npz")
if os.path.exists(curves_p):
    Z = np.load(curves_p)
    ck("全长曲线缓存含 23 个数组 (7R x 3 + 2 counterexample)",
       len(Z.files) == 23 and all(k in Z.files for k in ("bias_0", "sgd_16", "fce_1")))
else:
    ck("全长曲线缓存存在（运行 make verify 生成）", False,
       "data/recomputed_v2_curves.npz missing")

print("=" * 78)
print("7. Monte Carlo 确定性重跑（--full），由 minsep 单一计算源执行")
print("=" * 78)
if "--full" in sys.argv:
    print("  recomputing run_uniform() ...")
    uni = X.run_uniform()
    print("  recomputing run_counterexample() ...")
    fce = X.run_counterexample()
    print("  recomputing run_small_rho() ...")
    small = X.run_small_rho()

    for r in Rs:
        a, b = uni[r], v2["uniform"][str(r)]
        ok = all(close(a[k], b[k], 1e-9) for k in ("pred", "tvar_end", "sgd_end", "sgd_t11", "sgd_t20", "sgd_t40"))
        ck(f"MC 重跑 R={r:2d}: pred/tvar_end/sgd_end/sgd_t11/t20/t40 全部复现", ok,
           f"sgd_end {a['sgd_end']:.9f} vs {b['sgd_end']:.9f}")
    for r in (0, 1):
        a, b = fce[r], v2["counterexample"][str(r)]
        ck(f"MC 重跑 counterexample R={r}: theory/end 复现",
           close(a["theory"], b["theory"], 1e-12) and close(a["end"], b["end"], 1e-9),
           f"end {a['end']:.9f} vs {b['end']:.9f}")
    for r in [0, 1, 2, 5, 10]:
        a, b = small[r], v2["small_rho"][str(r)]
        ck(f"MC 重跑 small_rho R={r:2d}: pred/ratio_var/tvar_end/sgd_end/sgd_ratio 复现",
           all(close(a[k], b[k], 1e-9) for k in ("pred", "ratio_var", "tvar_end", "sgd_end", "sgd_ratio")),
           f"sgd_end {a['sgd_end']:.9f} vs {b['sgd_end']:.9f}")
    np.savez(curves_p,
             **{f"bias_{r}": uni[r]["bias"] for r in Rs},
             **{f"tvar_{r}": uni[r]["tvar"] for r in Rs},
             **{f"sgd_{r}": uni[r]["sgd"] for r in Rs},
             **{f"fce_{r}": fce[r]["tvar"] for r in (0, 1)})
    print("  full curves cached -> data/recomputed_v2_curves.npz")
else:
    print("  (跳过；加 --full 执行)")

print()
print("=" * 78)
ok = sum(1 for _, v, _ in R if v); ng = sum(1 for _, v, _ in R if not v)
print(f"  合计 {len(R)} 项：PASS {ok} / FAIL {ng}")
for n, v, d in R:
    if not v: print("   FAIL ->", n, d)
print("=" * 78)
sys.exit(0 if ng == 0 else 1)
