# -*- coding: utf-8 -*-
r"""figstyle.py --- 投稿图统一样式（JMLR 合规字号）。

画布宽度 6.0 inch = main.tex 中 \textwidth 的最终印刷宽度，因此 matplotlib
里设定的字号就是 PDF 中的实际字号，不再被缩小一半。高度 3.2 inch 多出的
部分专用于放置图例；缩放按宽度进行，字号不受高度影响。
所有文字（刻度、轴标签、面板标题、图例、总标题）>= 9.5 pt，满足 JMLR 对
illustration 文字至少 9 pt 的要求。

图例统一放在**每个子图正下方、横排**（一行放不下时自动折成多行），无边框，
不再占用绘图区，避免遮盖数据曲线。twinx 双轴的句柄合并进同一个图例。

输出为 600 dpi JPEG 位图。画布物理尺寸仍是 6.0 x 3.2 in，pdflatex 以
width=\textwidth 做 1:1 嵌入，因此 9.5 pt 的字在印刷件里仍是 9.5 pt
（600 dpi 下约 79 px 高）。所有字形在 Agg 端已栅格化为像素，最终 PDF 里
不再嵌套 CID 字体子集或透明组，避免阅读器丢失数学符号；JPEG 走 DCTDecode
直通嵌入，不会像 PNG 那样触发 pdfTeX 逐张重编码。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")

CANVAS = (6.0, 3.20)          # inch; width == \textwidth, tight height for legends
DPI = 600                    # raster resolution; physical size stays 6.0 x 3.2 in
PT = 9.5                     # minimum type size in the printed figure
ADJUST = dict(left=0.105, right=0.945, bottom=0.330, top=0.880, wspace=0.36)
LEGEND_GAP = 0.014           # figure-fraction gap below the lowest x-axis text


def apply_rc():
    matplotlib.rcParams.update({
        "font.size": PT,
        "axes.titlesize": PT + 0.5, "axes.labelsize": PT + 0.5,
        "xtick.labelsize": PT, "ytick.labelsize": PT,
        "legend.fontsize": PT,
        "lines.linewidth": 1.5, "lines.markersize": 4.0,
        "axes.grid": True, "grid.alpha": .30, "grid.linewidth": .6,
        "axes.titlepad": 4,
        "figure.dpi": DPI, "savefig.dpi": DPI,
        "savefig.bbox": None, "savefig.pad_inches": 0,
        "figure.facecolor": "white",
        # Sans-serif family is fixed so the same glyph outlines are used on
        # every machine; mathtext symbols are rasterised by Agg at 600 dpi.
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
    })


def new_canvas():
    import matplotlib.pyplot as plt
    return plt.subplots(1, 2, figsize=CANVAS, dpi=DPI)


def _bbox_figure(fig, bbox_display):
    return bbox_display.transformed(fig.transFigure.inverted())


def _headroom(ax, frac=0.08):
    """Small top margin so lines do not touch the panel title; no legend there."""
    try:
        y0, y1 = ax.get_ylim()
        if ax.get_yscale() == "log":
            if y0 > 0 and y1 > 0:
                ax.set_ylim(y0, y1 * 1.5)
        else:
            ax.set_ylim(y0, y1 + frac * (y1 - y0))
    except Exception:
        pass


def _x_text_boxes(fig, renderer):
    """Bboxes (figure coords) of x tick labels and x axis labels."""
    boxes = []
    inv = fig.transFigure.inverted()
    for ax in fig.axes:
        for artist in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            try:
                if not artist.get_text():
                    continue
                bb = artist.get_window_extent(renderer).transformed(inv)
                boxes.append(bb)
            except Exception:
                pass
    return boxes


def _overlap(a, b):
    dx = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    dy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    return dx * dy


def finalize(fig, suptitle=None, adjust=None):
    """Unified layout: one horizontal legend below each subplot group.

    Twin axes (``twinx``) share the same x position but carry their own
    labelled lines; their handles are merged into a single legend.  Every
    legend is a figure-level artist anchored below the lowest x-axis text of
    its subplot, so it never covers a data curve.  The largest column count
    that fits the subplot width is chosen (one horizontal row when possible,
    wrapped to multiple rows otherwise).
    """
    fig.subplots_adjust(**(adjust or ADJUST))
    if suptitle:
        fig.suptitle(suptitle, fontsize=PT + 1.0, y=0.975)
    for ax in fig.axes:
        _headroom(ax)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Group by left edge: twins share x0, so each group = one subplot.
    groups = {}
    for ax in fig.axes:
        g = round(ax.get_position().x0, 2)
        groups.setdefault(g, []).append(ax)

    # Collect handles before removing any in-axes legend.
    group_data = []
    for g, axes in sorted(groups.items()):
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles += list(h)
            labels += list(l)
        for ax in axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        group_data.append((axes[0], handles, labels))
    for leg in list(fig.legends):
        leg.remove()

    text_boxes = _x_text_boxes(fig, renderer)
    lowest_text = min((b.y0 for b in text_boxes), default=0.30)
    legend_top = lowest_text - LEGEND_GAP

    placed = []
    for host, handles, labels in group_data:
        if not labels:
            continue
        pos = host.get_position()
        cx = (pos.x0 + pos.x1) / 2.0
        # Legends sit below the axes, so they may be a little wider than the
        # subplot itself; 1.28x still leaves a clear gap between the two
        # subplots' legends.
        max_w = (pos.x1 - pos.x0) * 1.29
        best = None
        # Try every column count and keep the layout with the smallest penalty:
        # out-of-page, too wide, overlap, and (mildly) number of rows.  This
        # prefers one horizontal row but never falls back to a tall 1-column
        # legend that would run off the bottom of the page.
        for ncol in range(len(labels), 0, -1):
            leg = fig.legend(
                handles, labels, ncol=ncol,
                loc="upper center", bbox_to_anchor=(cx, legend_top),
                bbox_transform=fig.transFigure,
                frameon=False, fontsize=PT,
                handlelength=1.4, handletextpad=.45,
                columnspacing=1.0, labelspacing=.24, borderpad=0.0,
            )
            fig.canvas.draw()
            bb = _bbox_figure(fig, leg.get_window_extent(renderer))
            rows = int(np.ceil(len(labels) / ncol))
            hard = (max(0.0, 0.004 - bb.y0) + max(0.0, -bb.x0)
                    + max(0.0, bb.x1 - 1.0) + max(0.0, bb.width - max_w)
                    + sum(_overlap(bb, tb) for tb in text_boxes)
                    + sum(_overlap(bb, q) for q in placed))
            # Hard constraints first; then prefer the fewest rows, i.e. a
            # horizontal strip rather than a tall single-column stack.
            penalty = (10000 * hard + 25.0 * rows
                       + 0.2 * (len(labels) - ncol))
            if best is None or penalty < best[0]:
                best = (penalty, ncol, bb)
            leg.remove()
        _, ncol, bb = best
        leg = fig.legend(
            handles, labels, ncol=ncol,
            loc="upper center", bbox_to_anchor=(cx, legend_top),
            bbox_transform=fig.transFigure,
            frameon=False, fontsize=PT,
            handlelength=1.4, handletextpad=.45,
            columnspacing=1.0, labelspacing=.24, borderpad=0.0,
        )
        fig.canvas.draw()
        placed.append(_bbox_figure(fig, leg.get_window_extent(renderer)))
    fig.canvas.draw()
    return fig


def audit(fig):
    """Check figure-level legends: inside the page, below the axes, and clear
    of x-axis text and of each other.  Empty list means fully compliant."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    boxes = []
    text_boxes = _x_text_boxes(fig, r)
    for leg in fig.legends:
        bb = _bbox_figure(fig, leg.get_window_extent(r))
        boxes.append(bb)
        if bb.x0 < -1e-3 or bb.y0 < -1e-3 or bb.x1 > 1 + 1e-3 or bb.y1 > 1 + 1e-3:
            bad.append(("outside page", np.round([bb.x0, bb.y0, bb.x1, bb.y1], 3).tolist()))
        for ax in fig.axes:
            if bb.y1 > ax.get_position().y0 + 1e-3:
                bad.append(("overlaps plotting area", ax.get_title() or "axes"))
                break
        if any(_overlap(bb, t) > 1e-6 for t in text_boxes):
            bad.append(("overlaps x-axis text", np.round([bb.x0, bb.y0, bb.x1, bb.y1], 3).tolist()))
        if min(float(t.get_fontsize()) for t in leg.get_texts()) < 9.0:
            bad.append(("legend font below 9 pt", None))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _overlap(boxes[i], boxes[j]) > 1e-6:
                bad.append(("legends overlap", (i, j)))
    return bad


def save(fig, path):
    bad = audit(fig)
    if bad:
        print("  [figstyle] WARNING legend placement:", bad)
    # Deterministic JPEG: pinned quality/subsampling and no wall-clock metadata,
    # so `make figures` reproduces the released bytes exactly.  Explicit
    # ``format`` keeps this independent of the filename extension.
    fig.savefig(path, dpi=DPI, format="jpeg",
                pil_kwargs={"quality": 95, "subsampling": 0,
                            "optimize": True, "dpi": (DPI, DPI)})
    import matplotlib.pyplot as plt
    plt.close(fig)
    return bad
