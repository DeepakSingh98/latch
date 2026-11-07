"""Latch site figures in the plate style (plate.py is copied from deepaksingh98.github.io/style/).

Values are the published measurements; nothing is re-measured.
Run: python3 -B site/figures/make_figures.py
"""
import json
from pathlib import Path

from plate import COLORS, LEFT, RIGHT, Plate

HERE = Path(__file__).resolve().parent


def bars(p, x0, w, head, sub, vals, labels, fmt, base=556, top=292):
    """One small multiple: serif headline number, then two bars on the group's own scale."""
    p.text(x0, 180, head, size=34, cls="serif")
    for i, line in enumerate(sub):
        p.text(x0, 210 + i * 24, line, size=18, cls="mute")
    gap = 28
    bar_w = min(200, (w - gap - 20) / 2)
    tallest = max(vals)
    p.line(x0, base + 1, x0 + 2 * bar_w + gap, base + 1, "axis", 2)
    for i, (v, lbl, col) in enumerate(zip(vals, labels, ("orange", "base"))):
        h = (base - top) * v / tallest
        bx = x0 + i * (bar_w + gap)
        p.rect(bx, base - h, bar_w, h, col)
        p.text(bx + bar_w / 2, base - h - 12, fmt[i], size=18, cls="mono", anchor="middle")
        p.text(bx + bar_w / 2, base + 30, lbl, size=18, cls="mute", anchor="middle")


def fig_chart():
    p = Plate("Latch on H100", "Three separate comparisons against compiled PyTorch.",
              "Three separate comparisons against compiled PyTorch. TinyMoE: 857 versus 1432 training-update seconds "
              "to validation loss 3.30, one confirmation run. Small-batch updates: 13.63–13.77 versus 37.50–38.42 ms. "
              "GradLev: 53.98 versus 189.53 ms per block, with CUDA graphs in the control.")
    bars(p, LEFT, 352, "40% less time", ["update time to loss 3.30", "TinyMoE · batch 32 × 1,024"],
         [857, 1432], ["Latch", "PyTorch"], ["857 s", "1,432 s"])
    bars(p, 424, 352, "2.75× faster", ["training updates", "TinyMoE · batch 2 × 1,024"],
         [13.70, 37.96], ["Latch", "PyTorch"], ["13.6–13.8 ms", "37.5–38.4 ms"])
    bars(p, 806, 352, "3.5× faster", ["test-time adaptation", "GradLev · 1,024 steps"],
         [53.98, 189.53], ["Latch", "PyTorch"], ["54 ms", "190 ms"])
    p.note("Time-to-loss: one run. Separate scales. GradLev control uses CUDA graphs.")
    p.legend([("Latch", "orange"), ("Compiled PyTorch", "base")])
    p.save(HERE / "latch-chart.svg", png=HERE / "latch-chart.png")


def fig_diagram():
    p = Plate("1,024 learning steps. One kernel.", "",
              "Schematic of GradLev adapting GPT-2 small’s final MLP. Compiled PyTorch with CUDA graphs runs 18,176 "
              "kernels. Latch runs the adaptation loop in one kernel and retains the adapted MLP weights in shared "
              "memory. Other GPU-memory accesses remain. Not a timing scale.")
    p.text(LEFT, 158, "Compiled PyTorch + CUDA graphs", size=24)
    p.text(RIGHT, 158, "18,176 kernels", size=22, cls="mute mono", anchor="end")
    groups = [(LEFT, "Token 1"), (LEFT + 296, "Token 2"), (RIGHT - 216, "Token 1,024")]
    for x0, label in groups:
        for i in range(12):
            p.rect(x0 + i * 18, 182, 12, 72, "base")
        p.text(x0 + 105, 288, label, size=20, cls="mute", anchor="middle")
    p.text(760, 226, "···", size=44, cls="mute", anchor="middle")
    p.text(LEFT, 372, "Latch", size=24)
    p.text(RIGHT, 372, "1 kernel", size=22, cls="mono", anchor="end")
    p.rect(LEFT, 396, RIGHT - LEFT, 92, "orange")
    for x0, label in groups:
        p.text(x0 + 105, 450, label, size=22, cls="bgink", anchor="middle", weight=600)
    p.text(760, 450, "···", size=22, cls="bgink", anchor="middle", weight=600)
    p.text((LEFT + RIGHT) / 2, 530, "Adapted MLP weights stay on-chip", size=22, anchor="middle")
    p.note("GradLev · GPT-2 small’s final MLP · schematic")
    p.text(RIGHT, 642, "3.5× faster", size=30, cls="serif", anchor="end")
    p.save(HERE / "latch-diagram.svg", png=HERE / "latch-diagram.png")


def fig_speed():
    p = Plate("Nestor-100M on H100", "Complete training updates of 262,144 tokens on one H100.",
              "Complete training updates (262,144 tokens) on one H100. Latch 1,553 ms versus the authors' PyTorch "
              "3,280 ms (2.1x) and versus hand-tuned PyTorch 1,924 ms (1.24x).")
    bars(p, LEFT, 520, "2.1× faster", ["than the authors' code", "training update · 262k tokens"], [1552.6, 3280], ["Latch", "Authors'"],
         ["1,553 ms", "3,280 ms"])
    bars(p, 640, 520, "1.24× faster", ["than hand-tuned PyTorch", "training update · 262k tokens"], [1552.6, 1924], ["Latch", "Hand-tuned"],
         ["1,553 ms", "1,924 ms"])
    p.note("Same H100, same model math. Separate scales. Latch runs one generated program per microbatch.")
    p.legend([("Latch", "orange"), ("PyTorch", "base")])
    p.save(HERE / "nestor-speed.svg")


def fig_learning():
    d = json.loads((HERE / "nestor-learning-data.json").read_text())
    p = Plate("Same learning in half the time", "2.1× less time to the same training loss. Nestor-100M, "
              "19,080 updates, 5B tokens.",
              "Training loss over a full Nestor-100M run (19,080 updates, identical batches), timed at each "
              "implementation's measured speed on the same H100: Latch 8.2 h, the authors' PyTorch 17.4 h.")
    X0, X1, Y0, Y1 = 96, RIGHT, 150, 570
    xmax, ylo, yhi = 19.5, 3.25, 4.45
    px = lambda h: X0 + (X1 - X0) * h / xmax
    py = lambda v: Y1 - (Y1 - Y0) * (v - ylo) / (yhi - ylo)
    for v in (3.4, 3.8, 4.2):
        p.line(X0, py(v), X1, py(v), "rule", 1.5)
        p.text(X0 - 14, py(v) + 6, f"{v:.1f}", size=18, cls="mute", anchor="end")
    for h in (0, 4, 8, 12, 16):
        p.line(px(h), Y0, px(h), Y1, "rule", 1.5)
        p.text(px(h), Y1 + 28, f"{h} h" if h == 16 else f"{h}", size=18, cls="mute", anchor="middle")
    p.line(X0, Y1, X1, Y1, "axis", 2)
    p.add(f'<clipPath id="plot"><rect x="{X0}" y="{Y0}" width="{X1 - X0}" height="{Y1 - Y0}"/></clipPath>')
    for key, col in (("authors", "base"), ("latch", "orange")):
        pts = " ".join(f"{px(h):.1f},{py(v):.1f}" for h, v in d[key])
        p.add(f'<polyline points="{pts}" fill="none" stroke="{COLORS[col]}" stroke-width="3" '
              'stroke-linejoin="round" stroke-linecap="round" clip-path="url(#plot)"/>')
    for key, name, col in (("latch", "Latch", "orange"), ("authors", "PyTorch", "base")):
        h, v = d[key][-1]
        p.add(f'<rect x="{px(h) - 7:.1f}" y="{py(v) - 7:.1f}" width="14" height="14" fill="{COLORS[col]}"/>')
        p.text(px(h), py(v) - 44, name, size=20, anchor="middle")
        p.text(px(h), py(v) - 20, f"{h:.1f} h", size=18, cls="mute mono", anchor="middle")
    p.note("Training loss on identical batches, timed at each implementation's measured speed on the same H100.")
    p.legend([("Latch", "orange"), ("PyTorch", "base")])
    p.save(HERE / "nestor-learning.svg")


def fig_sources():
    rows = [("More memory loads in flight", 178), ("Single-pass attention backward", 52),
            ("Fused sibling matrix products", 49), ("Row fusion", 43), ("Optimizer and vocab head", 30)]
    p = Plate("Where the time went", "−345 ms per update in the generated program: 1,898 → 1,553 ms, approximate "
              "contributions.",
              "Approximate contributions to milliseconds saved per training update: "
              + ", ".join(f"{a} {b} ms" for a, b in rows)
              + ". The changes interact; the combined improvement from 1,898 to 1,553 ms is measured directly.")
    LX, BX, BW = 372, 396, 640
    p.line(BX, 172, BX, 568, "axis", 2)
    for i, (lbl, v) in enumerate(rows):
        y = 208 + i * 80
        p.text(LX, y + 8, lbl, size=20, anchor="end")
        p.rect(BX, y - 22, BW * v / 180, 44, "orange")
        p.text(BX + BW * v / 180 + 14, y + 8, f"−{v} ms", size=20, cls="mono")
    p.note("The changes interact. The combined 345 ms improvement is measured directly.")
    p.save(HERE / "nestor-sources.svg")


def fig_baselines():
    groups = [("Dense transformer", 3.578, [(9.24, "authors' code", "outline"), (4.23, "tuned", "base")]),
              ("MoE", 3.386, [(10.53, "authors' code", "outline"), (5.13, "tuned", "base")]),
              ("Nestor", 3.356, [(16.97, "authors' code", "outline"), (10.19, "tuned", "base"),
                                 (8.22, "Latch", "orange")])]
    p = Plate("GPU hours to train at 100M", "5B tokens on one H100. Held-out losses from the Nestor paper (Table 3); "
              "lower is better.",
              "Measured H100 hours to train each 100M model on 5B tokens, with held-out losses from the Nestor paper "
              "(Table 3). dense-authors 9.2 h; dense-tuned 4.2 h; moe-authors 10.5 h; moe-tuned 5.1 h; "
              "nestor-authors 17.0 h; nestor-tuned 10.2 h; latch 8.2 h.")
    LX, BX, H, GAP, GGAP = 268, 292, 36, 10, 44
    scale = 34.0
    y = 166
    for name, loss, rows in groups:
        top = y
        for hours, label, style in rows:
            w = hours * scale
            if style == "outline":
                p.frame(BX, y, w, H, "base", 2)
            else:
                p.rect(BX, y, w, H, style)
            p.text(BX + w + 12, y + 25, f"{hours:.1f} h", size=19, cls="mono")
            p.text(BX + w + 12 + 11.6 * len(f"{hours:.1f} h") + 14, y + 25, label, size=17, cls="mute")
            y += H + GAP
        mid = (top + y - GAP) / 2
        p.text(LX, mid - 2, name, size=22, cls="serif", anchor="end")
        p.text(LX, mid + 22, f"loss {loss:.3f}", size=17, cls="mute mono", anchor="end")
        y += GGAP - GAP
    p.legend([("Latch", "orange"), ("Tuned PyTorch", "base"), ("Authors' code", "outline")])
    p.save(HERE / "nestor-baselines.svg")


def curve(p, x1, y1, x2, y2, lift, stroke, width=1.6, opacity=1):
    mx = (x1 + x2) / 2
    p.add(f'<path d="M{x1:.1f} {y1:.1f} Q{mx:.1f} {y1 - lift:.1f} {x2:.1f} {y2:.1f}" fill="none" '
          f'stroke="{COLORS[stroke]}" stroke-width="{width}" opacity="{opacity:.2f}"/>')


def swatch_label(p, x, y, s, col):
    """Centred label with a square swatch: the colour names the category, the words stay in ink."""
    w = 9.2 * len(s)
    p.rect(x - w / 2 - 22, y - 13, 14, 14, col)
    p.text(x - w / 2, y, s, size=19)


def fig_structure():
    p = Plate("Nestor is lots of small, uneven work", "",
              "Schematic of one Nestor layer, repeated 13 times: mixing earlier layers, attention over uneven columns, "
              "and uneven experts. PyTorch runs each step as a separate kernel with its own trip to GPU memory. Latch "
              "fuses them into one program per microbatch, with reads batched and routing fused into its neighbours.")
    base = 256
    # mix earlier layers
    for i in range(7):
        x = 62 + i * 21
        p.rect(x, 146, 10, base - 146, "green", 0.25 + i * 0.125)
        curve(p, x + 5, 146, 290, 136, 34 + (6 - i) * 2, "green", 1.6, 0.25 + i * 0.125)
    p.rect(284, 146, 12, base - 146, "green")
    swatch_label(p, 176, 292, "mix earlier layers", "green")
    p.arrow(322, 201, 356, 201)
    # attention columns
    size, pitch = 10, 14
    for i in range(8):
        p.rect(392, base - 10 - i * pitch, size, size, "ink" if i == 7 else "base")
    heads = {}
    for x, n, head in ((500, 7, True), (548, 3, False), (596, 7, True), (644, 2, True)):
        p.frame(x - 5, base - (n - 1) * pitch - 15, size + 10, (n - 1) * pitch + 20, "axis", 1)
        for i in range(n):
            is_head = head and i == n - 1
            p.rect(x, base - 10 - i * pitch, size, size, "ink" if is_head else "blue")
        if head:
            heads[x] = base - 10 - (n - 1) * pitch - 7
    for x, top in heads.items():
        curve(p, 402, 158, x + 5, top - 8, 44, "base", 1.6)
        p.add(f'<polygon points="{x + 5:.1f},{top:.1f} {x:.1f},{top - 8:.1f} {x + 10:.1f},{top - 8:.1f}" '
              f'fill="{COLORS["base"]}"/>')
    swatch_label(p, 524, 292, "attention columns", "blue")
    p.arrow(700, 201, 734, 201)
    # experts
    for k, n in enumerate((5, 1, 3, 7, 2, 4, 1, 3)):
        x = 776 + k * 32
        p.frame(x - 5, base - (n - 1) * pitch - 15, size + 10, (n - 1) * pitch + 20, "axis", 1)
        for i in range(n):
            p.rect(x, base - 10 - i * pitch, size, size, "orange")
    swatch_label(p, 888, 292, "experts", "orange")
    p.text(RIGHT, 216, "×13", size=52, cls="serif", anchor="end")
    p.text(RIGHT, 246, "layers", size=19, cls="mute", anchor="end")
    p.line(LEFT, 318, RIGHT, 318, "axis", 1)

    # PyTorch: one kernel, one trip to memory
    L, R = 212, RIGHT
    p.text(LEFT, 380, "PyTorch", size=30, cls="serif")
    top, h = 350, 36
    x = L
    for w, col in ((46, "green"), (46, "green"), (46, "green"), (46, "green"), (56, "base"), (212, "blue"),
                   (56, "base"), (270, "orange"), (R - L - 778 - 8 * 11, "base")):
        p.rect(x, top, w, h, col)
        cx = x + w / 2
        p.arrow(cx - 5, top + h + 22, cx - 5, top + h + 4, "base", 1.5, 6)
        p.arrow(cx + 5, top + h + 4, cx + 5, top + h + 22, "base", 1.5, 6)
        x += w + 11
    p.rect(L, 414, R - L, 24, "cell")
    p.text(R - 12, 432, "GPU memory", size=15, cls="mute", anchor="end")

    # Latch: one program, fewer trips
    top = 508
    p.text(LEFT, 538, "Latch", size=30, cls="serif")
    p.frame(L - 7, top - 7, R - L + 14, h + 14, "ink", 2)
    x = L
    segs = ((208, "green", True), (62, "base", False), (226, "blue", True), (62, "base", False), (0, "orange", False))
    orange_w = R - L - 208 - 62 - 226 - 62 - 2 * 30
    for w, col, split in segs:
        w = w or orange_w - 62
        p.rect(x, top, w, h, col)
        x += w
        if split:
            p.line(x + 15, top - 2, x + 15, top + h + 2, "ink", 2, "2 4")
            x += 30
    p.rect(x, top, R - x, h, "base")
    p.text(L + 104, top - 18, "reads batched", size=18, cls="mute", anchor="middle")
    p.text(L + 208 + 30 + 240, top - 18, "routing fused into its neighbours", size=18, cls="mute", anchor="middle")
    for cx in (L + 104, L + 208 + 30 + 62 + 113, R - 62 - orange_w / 2):
        p.arrow(cx - 5, top + h + 30, cx - 5, top + h + 12, "base", 1.5, 6)
        p.arrow(cx + 5, top + h + 12, cx + 5, top + h + 30, "base", 1.5, 6)
    p.rect(L, 582, R - L, 24, "cell")
    p.text(R - 12, 600, "GPU memory", size=15, cls="mute", anchor="end")
    p.text(L + (R - L) / 2, 638, "one program per microbatch", size=19, anchor="middle")
    p.save(HERE / "nestor-structure.svg")


if __name__ == "__main__":
    fig_chart()
    fig_diagram()
    fig_speed()
    fig_learning()
    fig_sources()
    fig_baselines()
    fig_structure()
