"""Figure helper for the site style. See style/STYLE.md for the rules.

A plate is a 16:9 dark figure: serif title, muted one-line subtitle, flat marks, no panels.
Plates look the same in the light and dark page themes, so they are safe as <img> and as link previews.

  from plate import Plate
  p = Plate("Title", "One-line subtitle.", desc="Alt description with the numbers.")
  p.rect(100, 200, 300, 40, "orange")
  p.note("Short footnote.")
  p.legend([("Latch", "orange"), ("PyTorch", "base")])
  p.save("figures/name.svg", png="figures/name.png")

For matplotlib figures, call use_matplotlib() first and pass COLORS values to your marks.
"""
import subprocess
from pathlib import Path

W, H = 1200, 675
LEFT, RIGHT = 42, 1158
COLORS = {
    "bg": "#151513",
    "ink": "#ecebe5",
    "mute": "#9a988f",
    "base": "#77766e",  # baseline series: the thing being compared against
    "cell": "#2b2a26",  # empty cell, memory bars
    "rule": "#2a2925",  # gridlines
    "axis": "#55534c",
    "orange": "#ff8a3d",  # the subject of the figure
    "blue": "#45a8ff",  # second series, cost, negative
    "green": "#6fd08c",  # third series only
}
SERIF = '"Iowan Old Style",Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif'
SANS = '-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Helvetica,Arial,sans-serif'
MONO = 'ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace'
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
STYLE = (f"text{{font-family:{SANS};fill:{COLORS['ink']}}}"
         f".mute{{fill:{COLORS['mute']}}}.bgink{{fill:{COLORS['bg']}}}"
         f".serif{{font-family:{SERIF}}}.mono{{font-family:{MONO}}}")


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Plate:
    def __init__(self, title, subtitle="", desc=""):
        self.title, self.desc, self.body = title, desc or subtitle, []
        self.text(LEFT, 72, title, size=46, cls="serif")
        if subtitle:
            self.text(LEFT, 114, subtitle, size=20, cls="mute")

    def add(self, s):
        self.body.append(s)

    def text(self, x, y, s, size=18, cls="", anchor="start", weight=400):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" class="{cls}" '
                 f'text-anchor="{anchor}">{esc(s)}</text>')

    def rect(self, x, y, w, h, fill, opacity=1):
        op = f' opacity="{opacity:.2f}"' if opacity < 1 else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{COLORS[fill]}"{op}/>')

    def frame(self, x, y, w, h, stroke="ink", width=2):
        i = width / 2
        self.add(f'<rect x="{x + i:.1f}" y="{y + i:.1f}" width="{w - width:.1f}" height="{h - width:.1f}" fill="none" '
                 f'stroke="{COLORS[stroke]}" stroke-width="{width}"/>')

    def line(self, x1, y1, x2, y2, stroke="rule", width=1, dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{COLORS[stroke]}" '
                 f'stroke-width="{width}"{d}/>')

    def arrow(self, x1, y1, x2, y2, stroke="base", width=2, size=8):
        """Straight arrow; the head points along the axis of the longer side."""
        self.line(x1, y1, x2, y2, stroke, width)
        c = COLORS[stroke]
        if abs(x2 - x1) >= abs(y2 - y1):
            s = 1 if x2 > x1 else -1
            pts = f"{x2 + s * size:.1f},{y2:.1f} {x2:.1f},{y2 - size * .6:.1f} {x2:.1f},{y2 + size * .6:.1f}"
        else:
            s = 1 if y2 > y1 else -1
            pts = f"{x2:.1f},{y2 + s * size:.1f} {x2 - size * .6:.1f},{y2:.1f} {x2 + size * .6:.1f},{y2:.1f}"
        self.add(f'<polygon points="{pts}" fill="{c}"/>')

    def cells(self, x, y, rows, cols, on=None, size=12, gap=3):
        """A weight-style grid: dim cells, with on[(row, col)] = colour name for the bright ones."""
        on = on or {}
        for r in range(rows):
            for c in range(cols):
                self.rect(x + c * (size + gap), y + r * (size + gap), size, size, on.get((r, c), "cell"))

    def note(self, s):
        self.text(LEFT, 640, s, size=18, cls="mute")

    def legend(self, items, y=634):
        """Square swatches, right-aligned on the bottom line. items: (label, colour) or (label, 'outline')."""
        widths = [16 + 8 + 9.4 * len(lbl) for lbl, _ in items]
        x = RIGHT - sum(widths) - 28 * (len(items) - 1)
        for (lbl, col), w in zip(items, widths):
            if col == "outline":
                self.frame(x, y - 10, 16, 16, "base", 2)
            else:
                self.rect(x, y - 10, 16, 16, col)
            self.text(x + 24, y + 4, lbl, size=18, cls="mute")
            x += w + 28

    def render(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
                f'aria-labelledby="title desc">\n<title id="title">{esc(self.title)}</title>'
                f'<desc id="desc">{esc(self.desc)}</desc>\n<style>{STYLE}</style>\n'
                f'<rect width="{W}" height="{H}" fill="{COLORS["bg"]}"/>\n' + "\n".join(self.body) + "\n</svg>\n")

    def save(self, svg_path, png=None):
        svg_path = Path(svg_path)
        svg_path.write_text(self.render())
        if png:
            subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            f"--screenshot={Path(png).resolve()}", f"--window-size={W},{H}",
                            "--force-device-scale-factor=2", svg_path.resolve().as_uri()],
                           check=True, capture_output=True, timeout=50)


def use_matplotlib():
    """rcParams matching the plates: same palette, no spines on top or right, thin grid."""
    import matplotlib as mpl
    c = COLORS
    mpl.rcParams.update({
        "figure.facecolor": c["bg"], "axes.facecolor": c["bg"], "savefig.facecolor": c["bg"],
        "figure.figsize": (19.2, 10.8), "figure.dpi": 100, "savefig.dpi": 100,
        "text.color": c["ink"], "axes.labelcolor": c["mute"], "axes.edgecolor": c["axis"],
        "xtick.color": c["mute"], "ytick.color": c["mute"], "grid.color": c["rule"], "grid.linewidth": 1.5,
        "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif", "font.size": 20, "axes.titlesize": 26, "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 3,
        "axes.prop_cycle": mpl.cycler(color=[c["orange"], c["blue"], c["green"]]),
    })
