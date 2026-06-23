import hashlib
import io
import random
from collections.abc import Callable
from functools import partial, wraps
from typing import ParamSpec

import equinox as eqx
import jax.numpy as jnp
import manim as m
from manim_slides import Slide
import numpy as np
import softjax as sj
from jaxtyping import Array, Bool, Int
from manim_slides import Slide
from PIL import Image
import plotly.graph_objects as go
from plotly.colors import convert_to_RGB_255
from plotly.subplots import make_subplots

P = ParamSpec("P")

# Constants

TITLE_FONT_SIZE = 48
CONTENT_FONT_SIZE = 32
SOURCE_FONT_SIZE = 24

# Colors

BS_COLOR = m.BLUE_D
UE_COLOR = m.MAROON_D
SIGNAL_COLOR = m.BLUE_B
WALL_COLOR = m.LIGHT_BROWN
INVALID_COLOR = m.RED
VALID_COLOR = "#28C137"
IMAGE_COLOR = "#636463"
X_COLOR = m.DARK_BROWN

# Hard (discrete) vs. soft (differentiable) branch colors for the argmax slide.
HARD_GREEN = "#1A8A6E"
SOFT_MAGENTA = "#C0399A"
NEUTRAL_BAR = "#6C7A89"

# Soft relaxation modes -> colors (used in the heaviside / boolean slide).
# Every soft operator in SoftJAX exposes these continuity modes via `mode=`.
HARD_COLOR = m.BLACK
# Light grey background for `Code` mobjects (readable on the white slide).
CODE_BG_COLOR = "#E8E8E8"
SOFT_MODES = [
    ("smooth", m.BLUE_D),
    ("c0", m.GREEN_D),
    ("c1", m.ORANGE),
    ("c2", m.MAROON_D),
]

# Manim defaults

tex_template = m.TexTemplate()
tex_template.add_to_preamble(
    r"""
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{mathtools}
"""
)

m.MathTex.set_default(
    color=m.BLACK, tex_template=tex_template, font_size=CONTENT_FONT_SIZE
)
m.Tex.set_default(color=m.BLACK, tex_template=tex_template, font_size=CONTENT_FONT_SIZE)
m.Text.set_default(color=m.BLACK, font_size=CONTENT_FONT_SIZE)


def cleanup_figure(
    fig: go.Figure,
    *,
    width: int | None = None,
    height: int | None = None,
    margin: dict[str, int] | None = None,
    show_xaxes: bool = False,
    show_yaxes: bool = False,
    show_zaxes: bool = False,
) -> go.Figure:
    if margin is None:
        margin = dict(l=0, r=0, t=0, b=0)

    fig.update_layout(
        width=width,
        height=height,
        margin=margin,
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=show_xaxes, backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(visible=show_yaxes, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(visible=show_zaxes, backgroundcolor="rgba(0,0,0,0)"),
        )
    )

    return fig


def move_camera(
    fig: go.Figure,
    *,
    elevation: int | float = 0,
    azimuth: int | float = 0,
    distance: int | float = 10,
) -> go.Figure:
    x, y, z = spherical_to_cartesian(
        np.asarray([distance, elevation, azimuth])
    ).tolist()

    camera = dict(
        up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=x, y=y, z=z)
    )

    fig.update_scenes(camera=camera)

    return fig


def fig_to_mobject(
    func: Callable[P, go.Figure],
    width: int | None = None,
    height: int | None = None,
    scale: int | float | None = 2,
) -> m.ImageMobject | m.opengl.OpenGLImageMobject:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> m.ImageMobject:
        fig = func(*args, **kwargs)
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_arr = np.asarray(img_pil)
        return m.ImageMobject(img_arr)

    return wrapper


class Presentation(Slide):
    skip_reversing = True

    # ------------------------------------------------------------------ #
    # Slide management helpers (pattern from examples/main.py)
    # ------------------------------------------------------------------ #

    def _init_canvas(self):
        self.camera.background_color = m.WHITE
        self.wait_time_between_slides = 0.1

        self.slide_number = (
            m.Integer(number=0, font_size=SOURCE_FONT_SIZE, edge_to_fix=m.UR)
            .set_color(m.GRAY)
            .to_corner(m.DR)
        )
        # Plain section headings -> Text (no LaTeX needed; also avoids the
        # empty-Tex compile error that m.Tex("") would raise).
        self.slide_title = (
            m.Text("", font_size=TITLE_FONT_SIZE)
            .to_corner(m.UL)
        )
        self.add_to_canvas(
            slide_number=self.slide_number,
            slide_title=self.slide_title,
        )

    def _next_slide_number_animation(self):
        return self.slide_number.animate(run_time=0.3).increment_value(1)

    def _next_slide_title_animation(self, title: str):
        new_title = m.Text(title, font_size=TITLE_FONT_SIZE).to_corner(m.UL)
        return m.Transform(self.slide_title, new_title)

    def new_clean_slide(self, title: str, contents=None, **kwargs):
        """Wipe current content and transition to a new section."""
        if self.mobjects_without_canvas:
            self.play(
                self._next_slide_number_animation(),
                self._next_slide_title_animation(title),
                self.wipe(
                    self.mobjects_without_canvas,
                    contents if contents else [],
                    return_animation=True,
                    **kwargs,
                ),
            )
        else:
            self.play(
                self._next_slide_number_animation(),
                self._next_slide_title_animation(title),
            )

    # ------------------------------------------------------------------ #
    # Scene methods
    # ------------------------------------------------------------------ #

    def construct(self):
        self._init_canvas()
        self.title()
        self.next_slide()
        #self.intro()
        #self.next_slide()
        self.heaviside_and_bools()
        self.next_slide()
        self.comparisons_and_logic()
        self.next_slide()
        #self.relu()
        #self.next_slide()
        self.argmax()
        self.next_slide()
        #self.sort()
        #self.next_slide()
        #self.sorting_algos()
        #self.next_slide()
        #self.benchmarks()

    def title(self):
        # Title slide: plain text (no LaTeX needed) laid out with relative
        # positioning (arrange) so it stays centred regardless of edits.
        main = m.Text("SoftJAX & SoftTorch", weight=m.BOLD, font_size=TITLE_FONT_SIZE)
        subtitle = m.VGroup(
            m.Text("Empowering Automatic Differentiation", font_size=CONTENT_FONT_SIZE),
            m.Text("Libraries with Informative Gradients", font_size=CONTENT_FONT_SIZE),
        ).arrange(m.DOWN, buff=0.15)
        header = m.VGroup(main, subtitle).arrange(m.DOWN, buff=0.35)

        presenter = m.Text(
            "A. René Geist - June 2026, AI4PEX", slant=m.ITALIC, font_size=CONTENT_FONT_SIZE-5
        )
        authors = m.Text(
            "Authors:   Anselm Paulus*   A. René Geist*   Vít Musil   Sebastian Hoffmann   Georg Martius",
            font_size=CONTENT_FONT_SIZE-15,
        )
        credits = m.VGroup(presenter, authors).arrange(m.DOWN, buff=0.25)

        title_group = m.VGroup(header, credits).arrange(m.DOWN, buff=0.9).move_to(m.ORIGIN)

        # Defensive: shrink to fit if a name list ever overflows the frame.
        max_width = m.config.frame_width - 1.0
        if title_group.width > max_width:
            title_group.scale(max_width / title_group.width)

        # Logos — SVGMobject for vector, ImageMobject for raster.
        # ImageMobjects are not VMobjects so the outer container is m.Group.
        icml_logo = (
            m.SVGMobject("images/logos/ICML_logo_2026.svg", height=0.9)
            .to_corner(m.UR, buff=0.3)
        )
        ut_logo = (
            m.ImageMobject("images/logos/UT_Logo_hires.png")
            .set_height(0.7)
        )
        mpi_logo = (
            m.ImageMobject("images/logos/MPG_IS_Logo_RGB_grey_dark-ENG.png")
            .set_height(0.7)
        )
        inst_logos = (
            m.Group(ut_logo, mpi_logo)
            .arrange(m.RIGHT, buff=0.3)
            .to_corner(m.UL, buff=0.3)
        )

        self.play(
            m.FadeIn(m.Group(title_group, icml_logo, inst_logos), shift=0.3 * m.UP)
        )

    def intro(self):
        pass

    # ------------------------------------------------------------------ #
    # Unified plotting of SoftJAX elementwise operators
    # ------------------------------------------------------------------ #

    def _eval(self, fn, xs, *, mode, softness=None):
        """Evaluate a SoftJAX operator on a grid using the library directly."""
        if softness is None:
            ys = fn(jnp.asarray(xs), mode=mode)
        else:
            ys = fn(jnp.asarray(xs), softness=float(max(softness, 1e-3)), mode=mode)
        return np.asarray(ys)

    def _curve(self, axes, fn, *, mode, color, softness=None, width=3, n=400):
        """Single unified curve builder: SoftJAX op -> Manim VMobject.

        Used for every panel and every mode (hard / smooth / c0 / c1 / c2),
        so all functions are plotted through exactly one code path.
        """
        x0, x1 = axes.x_range[0], axes.x_range[1]
        y0, y1 = axes.y_range[0], axes.y_range[1]
        xs = np.linspace(x0, x1, n)
        ys = np.clip(self._eval(fn, xs, mode=mode, softness=softness), y0, y1)
        pts = [axes.c2p(float(x), float(y)) for x, y in zip(xs, ys)]
        return (
            m.VMobject()
            .set_points_as_corners(pts)
            .set_stroke(color=color, width=width)
        )

    def _y_ticks(self, axes, tick_values=(-1, 1)):
        """Tick marks + labels at the given y values, only if within y range."""
        y0, y1 = axes.y_range[0], axes.y_range[1]
        group = m.VGroup()
        for v in tick_values:
            if y0 < v < y1:
                center = axes.c2p(0, float(v))
                tick = m.Line(
                    center + 0.13 * m.LEFT, center + 0.13 * m.RIGHT,
                    color=m.GREY, stroke_width=2,
                )
                label = m.Text(
                    str(int(v)), font_size=18, color=m.GREY
                ).next_to(tick, m.LEFT, buff=0.08)
                group.add(tick, label)
        return group

    def _legend(self, modes=SOFT_MODES):
        # No "hard" entry: the hard reference is drawn but never labelled.
        items = m.VGroup()
        for name, color in modes:
            swatch = m.Line(m.ORIGIN, 0.45 * m.RIGHT, color=color, stroke_width=4)
            label = m.Text(name, font_size=22, color=m.BLACK)
            items.add(m.VGroup(swatch, label).arrange(m.RIGHT, buff=0.12))
        return items.arrange(m.RIGHT, buff=0.45)

    def _slider(self, tau, vmin=0.0, vmax=1.0, length=5.5):
        """A horizontal 'softness' slider whose handle tracks `tau`."""
        track = m.Line(
            length / 2 * m.LEFT, length / 2 * m.RIGHT, color=m.GREY, stroke_width=5
        )
        end_ticks = m.VGroup(
            m.Line(0.12 * m.DOWN, 0.12 * m.UP, color=m.GREY, stroke_width=3).move_to(
                track.get_start()
            ),
            m.Line(0.12 * m.DOWN, 0.12 * m.UP, color=m.GREY, stroke_width=3).move_to(
                track.get_end()
            ),
        )
        end_labels = m.VGroup(
            m.Text(f"{vmin:.1f}", font_size=22, color=m.BLACK).next_to(
                track.get_start(), m.DOWN, buff=0.15
            ),
            m.Text(f"{vmax:.1f}", font_size=22, color=m.BLACK).next_to(
                track.get_end(), m.DOWN, buff=0.15
            ),
        )
        name = m.Text("softness", font_size=28, color=m.BLACK).next_to(
            track, m.LEFT, buff=0.4
        )
        handle = m.Dot(radius=0.13, color=m.BLACK)

        def _prop():
            return float(np.clip((tau.get_value() - vmin) / (vmax - vmin), 0.0, 1.0))

        handle.add_updater(lambda d: d.move_to(track.point_from_proportion(_prop())))
        value = m.DecimalNumber(
            tau.get_value(), num_decimal_places=1, font_size=26, color=m.BLACK
        )

        def _update_value(d):
            d.set_value(tau.get_value())
            d.next_to(handle, m.UP, buff=0.18)

        value.add_updater(_update_value)
        return m.VGroup(track, end_ticks, end_labels, name, handle, value)

    def heaviside_and_bools(self):
        self.new_clean_slide("Softening")
        tau = m.ValueTracker(0.1)
        smooth_color = SOFT_MODES[0][1]

        # ---- Top row: heaviside formula (left) + heaviside plot (right) ----
        heav_axes = m.Axes(
            x_range=[-1.1, 1.1, 1],
            y_range=[-0.25, 1.25, 0.5],
            x_length=3.9,
            y_length=2.5,
            axis_config=dict(color=m.GREY, stroke_width=2, include_ticks=False),
            tips=False,
        )
        formula_h = m.MathTex(
            r"H(x)=\begin{cases}0 & x<0\\[2pt] 0.5 & x=0\\[2pt] 1 & x>0\end{cases}",
            font_size=38,
        )
        top = m.VGroup(formula_h, heav_axes).arrange(m.RIGHT, buff=1.3).to_edge(
            m.UP, buff=1.0
        )

        # ---- Bottom row: sign / round / abs plots --------------------------
        bottom_specs = [
            (sj.sign, "sign", [-1.1, 1.1, 1], [-1.1, 1.1, 1]),
            (sj.round, "round", [-1.1, 1.1, 1], [-1.1, 1.1, 1]),
            (sj.abs, "abs", [-1.1, 1.1, 1], [-0.25, 1.1, 1]),
        ]
        bottom_panels = []
        for fn, name, xr, yr in bottom_specs:
            axes = m.Axes(
                x_range=xr,
                y_range=yr,
                x_length=3.3,
                y_length=1.95,
                axis_config=dict(color=m.GREY, stroke_width=2, include_ticks=False),
                tips=False,
            )
            bottom_panels.append(dict(fn=fn, name=name, axes=axes, soft={}))
        bottom = (
            m.VGroup(*[p["axes"] for p in bottom_panels])
            .arrange(m.RIGHT, buff=0.7)
            .next_to(top, m.DOWN, buff=0.55)
        )
        for p in bottom_panels:
            p["lab"] = m.Text(p["name"], font_size=26, color=m.BLACK).next_to(
                p["axes"], m.UP, buff=0.1
            )

        legend = self._legend().next_to(bottom, m.DOWN, buff=0.45)
        slider = self._slider(tau).to_edge(m.DOWN, buff=0.35)

        # ================= Beat 1: heaviside formula + hard plot ============
        heav_hard = self._curve(
            heav_axes, sj.heaviside, mode="hard", color=HARD_COLOR, width=4
        )
        heav_ticks = self._y_ticks(heav_axes)
        self.play(m.Write(formula_h))
        self.play(m.Create(heav_axes), m.FadeIn(heav_ticks))
        self.play(m.Create(heav_hard))
        self.next_slide()

        # ===== Beat 2: slider + smooth curve + swap to sigmoid formula ======
        heav_smooth = self._curve(
            heav_axes, sj.heaviside, mode="smooth", color=smooth_color,
            softness=tau.get_value(),
        )
        formula_s = m.MathTex(
            r"H_\tau(x)=\dfrac{1}{1+e^{-x/\tau}}", font_size=42
        ).move_to(formula_h)
        self.play(m.FadeIn(slider, shift=0.2 * m.UP))
        self.play(
            m.Create(heav_smooth),
            m.ReplacementTransform(formula_h, formula_s),
        )
        self.next_slide()

        # ===== Beat 3: animate softness 0.1 -> 1.0 -> 0.0 -> 0.1 ============
        def _retrace(mobj):
            mobj.become(
                self._curve(
                    heav_axes, sj.heaviside, mode="smooth", color=smooth_color,
                    softness=tau.get_value(),
                )
            )

        heav_smooth.add_updater(_retrace)
        self.play(tau.animate.set_value(1.0), run_time=2.5, rate_func=m.linear)
        self.play(tau.animate.set_value(0.04), run_time=7.0, rate_func=m.linear)
        self.play(tau.animate.set_value(0.1), run_time=2.5, rate_func=m.linear)
        heav_smooth.clear_updaters()
        self.next_slide()

        # ===== Beat 4: add sign/round/abs plots (smooth, softness=0.1) ======
        for p in bottom_panels:
            p["ticks"] = self._y_ticks(p["axes"])
        self.play(
            m.Create(m.VGroup(*[p["axes"] for p in bottom_panels])),
            m.FadeIn(m.VGroup(*[p["lab"] for p in bottom_panels])),
            m.FadeIn(m.VGroup(*[p["ticks"] for p in bottom_panels])),
        )
        for p in bottom_panels:
            p["soft"]["smooth"] = self._curve(
                p["axes"], p["fn"], mode="smooth", color=smooth_color,
                softness=0.1,
            )
        self.play(*[m.Create(p["soft"]["smooth"]) for p in bottom_panels])
        self.next_slide()

        # ===== Beat 5: add c0 / c1 / c2 to sign/round/abs + legend ==========
        new_curves = []
        for p in bottom_panels:
            for mode, color in SOFT_MODES[1:]:
                cv = self._curve(
                    p["axes"], p["fn"], mode=mode, color=color, softness=0.1
                )
                p["soft"][mode] = cv
                new_curves.append(cv)
        self.play(m.FadeIn(legend), *[m.Create(c) for c in new_curves], run_time=1.5)

    def comparisons_and_logic(self):
        self.new_clean_slide("Fuzzy logic")
        smooth_color = SOFT_MODES[0][1]
        tau = m.ValueTracker(0.01)

        # Binary comparisons f(x, y): we fix the threshold y and sweep x, so a
        # single 1-D curve goes through the same unified `_curve` code path.
        y_val = 1.0

        def fix_y(fn):
            return lambda xs, **kw: fn(xs, y_val, **kw)

        specs = [
            (sj.greater, "greater"),
            (sj.less_equal, "less_equal"),
            (sj.isclose, "isclose"),
        ]

        panels = []
        for fn, name in specs:
            axes = m.Axes(
                x_range=[0.0, 2.0, 1.0],
                y_range=[-0.25, 1.25, 1.0],
                x_length=3.6,
                y_length=2.3,
                axis_config=dict(color=m.GREY, stroke_width=2, include_ticks=False),
                tips=False,
            )
            panels.append(dict(fn=fn, name=name, axes=axes))

        row = (
            m.VGroup(*[p["axes"] for p in panels])
            .arrange(m.RIGHT, buff=0.9)
            .move_to(0.5 * m.UP)
        )

        for p in panels:
            axes = p["axes"]
            # Function-name label above each plot via the Code class.
            p["code"] = (
                m.Code(
                    code_string=f"sj.{p['name']}(x, y)",
                    language="python",
                    add_line_numbers=False,
                    formatter_style="default",
                    background_config=dict(fill_color=CODE_BG_COLOR, stroke_width=0),
                )
                .scale(0.55)
                .next_to(axes, m.UP, buff=0.3)
            )
            # "x" axis label at the right tip of the x-axis.
            p["xlab"] = m.MathTex("x", font_size=32, color=m.BLACK).next_to(
                axes.x_axis.get_end(), m.RIGHT, buff=0.15
            )
            # "y" tick on the x-axis marking the threshold position.
            center = axes.c2p(y_val, 0)
            ytick = m.Line(
                center + 0.1 * m.DOWN, center + 0.1 * m.UP,
                color=m.GREY, stroke_width=2,
            )
            ylabel = m.MathTex("y", font_size=30, color=m.BLACK).next_to(
                ytick, m.DOWN, buff=0.12
            )
            p["ymark"] = m.VGroup(ytick, ylabel)
            # Dashed grey reference line at the threshold x = y.
            p["vline"] = m.DashedLine(
                axes.c2p(y_val, 0.0),
                axes.c2p(y_val, 1.2),
                color=m.GREY,
                stroke_width=2,
                dash_length=0.09,
            )
            # y-axis ticks (here only "1" lies within the [-0.25, 1.25] range).
            p["yticks"] = self._y_ticks(axes)

        # The same softness slider as slide 1, shown directly (no fade) and
        # starting at softness = 0.001 (nearly a hard step).
        slider = self._slider(tau).to_edge(m.DOWN, buff=0.4)
        self.add(slider)

        # ---- Beat 1: code labels, axes, and annotations ----
        self.play(
            m.LaggedStart(
                *[m.FadeIn(p["code"], shift=0.2 * m.DOWN) for p in panels],
                lag_ratio=0.25,
            )
        )
        self.play(
            m.Create(m.VGroup(*[p["axes"] for p in panels])),
            m.FadeIn(m.VGroup(*[p["xlab"] for p in panels])),
            m.FadeIn(m.VGroup(*[p["ymark"] for p in panels])),
            m.FadeIn(m.VGroup(*[p["yticks"] for p in panels])),
            m.FadeIn(m.VGroup(*[p["vline"] for p in panels])),
        )
        self.next_slide()

        # ---- Beat 2: draw smooth curves one panel at a time, left -> right ----
        for p in panels:
            p["curve"] = self._curve(
                p["axes"], fix_y(p["fn"]), mode="smooth",
                color=smooth_color, softness=tau.get_value(),
            )
        self.play(
            m.Succession(
                *[m.Create(p["curve"], rate_func=m.linear) for p in panels],
            ),
            run_time=3.0,
        )
        self.next_slide()

        # ---- Beat 3: soften — slide the handle to 0.5; curves track live ----
        def make_retrace(panel):
            def _retrace(mob):
                mob.become(
                    self._curve(
                        panel["axes"], fix_y(panel["fn"]), mode="smooth",
                        color=smooth_color, softness=tau.get_value(),
                    )
                )
            return _retrace

        for p in panels:
            p["curve"].add_updater(make_retrace(p))
        self.play(tau.animate.set_value(0.1), run_time=2.5, rate_func=m.linear)
        for p in panels:
            p["curve"].clear_updaters()
        self.next_slide()

        # ---- Beat 4: caption describing the smooth curve as a CDF ----
        caption = m.MarkupText(
            '<b>CDF</b> defining the probability that a Boolean '
            "operator equates to true.",
            font_size=30,
            color=m.BLACK,
        ).next_to(row, m.DOWN, buff=0.6)
        self.play(m.FadeIn(caption, shift=0.2 * m.UP))
        self.next_slide()

        # ---- Beat 5: drop the slider, reveal the fuzzy-logic operators ----
        logic_formulas = m.VGroup(
            m.MathTex(r"\mathrm{all}(p_1,...,p_n) = \prod_i p_i", font_size=34),
            m.MathTex(r"\mathrm{not}(p) = 1 - p", font_size=34),
        ).arrange(m.DOWN, buff=0.4, aligned_edge=m.LEFT)

        arrow = m.Arrow(
            m.ORIGIN, 1.1 * m.RIGHT, buff=0.0, color=m.BLACK, stroke_width=4
        )
        op_list = m.VGroup(
            *[m.Tex(name, font_size=34) for name in ("any", "and", "or", "xor")]
        ).arrange(m.DOWN, buff=0.28, aligned_edge=m.LEFT)

        selection = m.VGroup(
            m.Tex("Selection = Expectation", font_size=34),
            m.MathTex(
                r"z_i = p_i \cdot x_i + (1 - p_i) \cdot y_i", font_size=34
            ),
        ).arrange(m.DOWN, buff=0.3)

        composition = m.VGroup(logic_formulas, arrow, op_list, selection).arrange(
            m.RIGHT, buff=0.6
        )
        max_width = m.config.frame_width - 1.0
        if composition.width > max_width:
            composition.scale(max_width / composition.width)
        composition.move_to(2.3 * m.DOWN)

        # Nudge the selection block to the right and frame it in a soft box.
        selection.shift(0.4 * m.RIGHT)
        selection_box = m.SurroundingRectangle(
            selection,
            buff=0.3,
            corner_radius=0.15,
            color=SOFT_MODES[0][1],
            stroke_width=2.5,
        )

        self.play(m.FadeOut(slider, shift=0.3 * m.DOWN))
        self.play(m.FadeOut(caption, shift=0.3 * m.DOWN))
        self.play(m.Write(logic_formulas))
        self.next_slide()

        # Arrow -> list of the remaining fuzzy-logic operators.
        self.play(m.GrowArrow(arrow), m.Write(op_list))
        self.next_slide()

        # Selection as an expectation over a Bernoulli choice.
        self.play(m.Write(selection), m.Create(selection_box))

    # ------------------------------------------------------------------ #
    # Hard vs. soft argmax & indexing
    # ------------------------------------------------------------------ #

    def _bar(self, value, x, baseline_y, color, *, unit, width, opacity=1.0):
        """A single value-bar growing upward from `baseline_y` at position `x`."""
        h = max(float(value) * unit, 1e-3)
        bar = m.Rectangle(
            width=width,
            height=h,
            fill_color=color,
            fill_opacity=opacity,
            stroke_color=color,
            stroke_width=2,
        )
        bar.move_to([x, baseline_y + h / 2.0, 0.0])
        return bar

    def argmax(self):
        self.new_clean_slide("Indexing an array")

        # --- Values and the SoftJAX-computed soft selection (no reimplementation) ---
        x_vals = [0.1, 0.4, 0.8]
        arr = jnp.asarray(x_vals)
        soft_index = sj.argmax(arr, axis=0, softness=0.1, mode="smooth")
        weights = [float(w) for w in np.asarray(soft_index)]
        y_soft = float(sj.dynamic_index_in_dim(arr, soft_index, axis=0, keepdims=False))
        hard_idx = int(jnp.argmax(arr))
        y_hard = x_vals[hard_idx]

        unit = 3.2
        bar_w = 0.85
        baseline_y = -1.2
        xs = [-2.4, 0.0, 2.4]
        res_x = 3.9  # where plucked / weighted results land

        def bar(value, x, color, opacity=1.0):
            return self._bar(
                value, x, baseline_y, color, unit=unit, width=bar_w, opacity=opacity
            )

        # ------------------------- Scene 1: the array ------------------------- #
        axis = m.Line(
            [-3.6, baseline_y, 0], [3.6, baseline_y, 0], color=m.GREY, stroke_width=3
        )
        bars = [bar(v, x, NEUTRAL_BAR) for v, x in zip(x_vals, xs)]
        val_labels = [
            m.Text(f"{v:.1f}", font="monospace", font_size=26, color=m.BLACK).next_to(
                b, m.UP, buff=0.15
            )
            for v, b in zip(x_vals, bars)
        ]
        idx_labels = [
            m.Text(str(i), font="monospace", font_size=26, color=m.GREY).move_to(
                [x, baseline_y - 0.4, 0]
            )
            for i, x in enumerate(xs)
        ]
        arr_label = m.Text(
            "x = [0.1, 0.4, 0.8]", font="monospace", font_size=30, color=m.BLACK
        ).move_to([0, baseline_y - 1.15, 0])

        self.play(m.Create(axis))
        self.play(
            m.LaggedStart(
                *[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.3
            ),
            m.LaggedStart(
                *[m.FadeIn(l, shift=0.2 * m.UP) for l in val_labels], lag_ratio=0.3
            ),
            run_time=1.6,
        )
        self.play(m.FadeIn(m.VGroup(*idx_labels)), m.FadeIn(arr_label))
        self.next_slide()

        # ------------------------- Scene 2: hard argmax ----------------------- #
        self.play(self._next_slide_title_animation("Hard argmax"))

        spot = m.SurroundingRectangle(
            bars[0], color=HARD_GREEN, stroke_width=5, buff=0.12, corner_radius=0.08
        )
        self.play(m.Create(spot), run_time=0.4)
        for i in (1, hard_idx):
            self.play(
                m.Transform(
                    spot,
                    m.SurroundingRectangle(
                        bars[i],
                        color=HARD_GREEN,
                        stroke_width=5,
                        buff=0.12,
                        corner_radius=0.08,
                    ),
                ),
                run_time=0.35,
            )

        argmax_lbl = m.Tex(
            r"argmax $\rightarrow$ 2", color=HARD_GREEN, font_size=40
        ).move_to([0, 2.4, 0])
        # Snap: pick wins green, the others dim to grey.
        self.play(
            bars[hard_idx].animate.set_fill(HARD_GREEN, 1.0).set_stroke(HARD_GREEN),
            val_labels[hard_idx].animate.set_color(HARD_GREEN),
            *[
                bars[i].animate.set_fill(m.GREY, 0.35).set_stroke(m.GREY)
                for i in range(len(bars))
                if i != hard_idx
            ],
            *[
                val_labels[i].animate.set_color(m.GREY)
                for i in range(len(bars))
                if i != hard_idx
            ],
            m.FadeIn(argmax_lbl, shift=0.2 * m.DOWN),
        )

        # Pluck bar 2 out via dynamic_index_in_dim -> result y = 0.8.
        pick = bars[hard_idx].copy()
        hard_result = bar(y_hard, res_x, HARD_GREEN)
        # Arc over the bars so the label never overlaps them.
        pluck_arrow = m.CurvedArrow(
            bars[hard_idx].get_top() + 0.1 * m.UP,
            hard_result.get_top() + 0.1 * m.UP,
            color=HARD_GREEN,
            stroke_width=4,
            angle=-m.TAU / 6,
        )
        dyn_lbl = m.Text(
            "dynamic_index_in_dim", font="monospace", font_size=22, color=HARD_GREEN
        ).move_to([3.1, 2.05, 0])
        hard_y_label = m.Text(
            "y = 0.8", font="monospace", font_size=28, color=HARD_GREEN
        ).next_to(hard_result, m.DOWN, buff=0.25)

        self.play(m.Create(pluck_arrow), m.FadeIn(dyn_lbl))
        self.play(pick.animate.move_to(hard_result), run_time=0.9)
        self.play(m.FadeIn(hard_y_label))

        # "not differentiable" with a struck-through nabla.
        nabla = m.MathTex(r"\nabla", color=m.RED, font_size=40)
        strike = m.Line(
            nabla.get_corner(m.DL), nabla.get_corner(m.UR), color=m.RED, stroke_width=4
        )
        not_diff = m.Text("not differentiable", font_size=24, color=m.RED)
        hard_cap = m.VGroup(m.VGroup(nabla, strike), not_diff).arrange(
            m.RIGHT, buff=0.2
        ).next_to(hard_y_label, m.DOWN, buff=0.3)
        self.play(m.Write(not_diff), m.Create(m.VGroup(nabla, strike)))

        hard_only = m.VGroup(
            spot, argmax_lbl, pick, pluck_arrow, dyn_lbl, hard_y_label, hard_cap
        )
        self.next_slide()

        # ------------------------- Scene 3: soft argmax ----------------------- #
        self.play(
            m.FadeOut(hard_only),
            self._next_slide_title_animation("Soft argmax"),
            *[
                bars[i].animate.set_fill(NEUTRAL_BAR, 1.0).set_stroke(NEUTRAL_BAR)
                for i in range(len(bars))
            ],
            *[val_labels[i].animate.set_color(m.BLACK) for i in range(len(bars))],
        )

        # Translucent magenta probability mass on each bar (opacity == weight).
        overlays = [
            bar(v, x, SOFT_MAGENTA, opacity=w)
            for v, x, w in zip(x_vals, xs, weights)
        ]
        w_labels = [
            m.Text(f"{w:.3f}", font="monospace", font_size=24, color=SOFT_MAGENTA)
            .next_to(val_labels[i], m.UP, buff=0.18)
            for i, w in enumerate(weights)
        ]

        sigma_tr = m.ValueTracker(0.0)
        sigma_num = m.DecimalNumber(
            0.0, num_decimal_places=1, font_size=34, color=SOFT_MAGENTA
        )
        sigma_num.add_updater(lambda d: d.set_value(sigma_tr.get_value()))
        sigma_lbl = m.MathTex(r"\sum_i w_i =", color=SOFT_MAGENTA, font_size=38)
        sigma = m.VGroup(sigma_lbl, sigma_num).arrange(m.RIGHT, buff=0.2).move_to(
            [0, 2.5, 0]
        )
        # Add the live counter so its updater runs while the mass accumulates.
        self.add(sigma_num)

        self.play(
            m.LaggedStart(*[m.FadeIn(o) for o in overlays], lag_ratio=0.2),
            m.LaggedStart(
                *[m.FadeIn(w, shift=0.2 * m.UP) for w in w_labels], lag_ratio=0.2
            ),
            m.FadeIn(sigma_lbl),
            sigma_tr.animate.set_value(1.0),
            run_time=2.2,
        )
        sigma_num.clear_updaters()
        self.next_slide()

        # ------------------------ Scene 4: weighted sum ----------------------- #
        terms = m.VGroup(
            m.MathTex(r"0.1\cdot 0.004", font_size=30),
            m.MathTex("+", font_size=30),
            m.MathTex(r"0.4\cdot 0.042", font_size=30),
            m.MathTex("+", font_size=30),
            m.MathTex(r"0.8\cdot 0.953", font_size=30),
            m.MathTex("=", font_size=30),
            m.MathTex(f"{y_soft:.2f}", color=SOFT_MAGENTA, font_size=34),
        ).arrange(m.RIGHT, buff=0.18).move_to([-0.7, baseline_y - 1.15, 0])

        soft_result = bar(y_soft, res_x, SOFT_MAGENTA)
        soft_y_label = m.Text(
            f"y = {y_soft:.2f}", font="monospace", font_size=28, color=SOFT_MAGENTA
        ).next_to(soft_result, m.UP, buff=0.15)
        expected_lbl = m.Text(
            "(expected value)", font_size=22, color=SOFT_MAGENTA
        ).next_to(soft_result, m.DOWN, buff=0.25)

        self.play(m.FadeOut(arr_label), m.Write(terms), run_time=1.4)
        self.play(
            m.GrowFromEdge(soft_result, m.DOWN),
            m.FadeIn(soft_y_label),
            m.FadeIn(expected_lbl),
        )
        self.next_slide()

        # --------------------- Scene 5: side-by-side payoff ------------------- #
        array_grp = m.VGroup(
            axis,
            *bars,
            *val_labels,
            *idx_labels,
            *overlays,
            *w_labels,
            sigma,
            terms,
            soft_result,
            soft_y_label,
            expected_lbl,
        )
        self.play(
            m.FadeOut(array_grp), self._next_slide_title_animation("Hard vs. soft")
        )

        cmp_base = -1.4
        # Left: hard / discrete (green).
        h_bar = self._bar(y_hard, -3.0, cmp_base, HARD_GREEN, unit=unit, width=1.0)
        h_title = m.Text("Hard", font_size=30, weight=m.BOLD, color=HARD_GREEN).move_to(
            [-3.0, 2.3, 0]
        )
        h_val = m.Text(
            "y = 0.8", font="monospace", font_size=28, color=HARD_GREEN
        ).next_to(h_bar, m.UP, buff=0.2)
        h_sub = m.Text("discrete", font_size=24, color=m.GREY).next_to(
            h_bar, m.DOWN, buff=0.4
        )
        h_nabla = m.VGroup(
            m.MathTex(r"\nabla", color=m.RED, font_size=44),
        )
        h_strike = m.Line(
            h_nabla.get_corner(m.DL),
            h_nabla.get_corner(m.UR),
            color=m.RED,
            stroke_width=4,
        )
        h_grad = m.VGroup(h_nabla, h_strike).next_to(h_sub, m.DOWN, buff=0.35)

        # Right: soft / weighted (magenta).
        s_bar = self._bar(y_soft, 3.0, cmp_base, SOFT_MAGENTA, unit=unit, width=1.0)
        s_title = m.Text(
            "Soft", font_size=30, weight=m.BOLD, color=SOFT_MAGENTA
        ).move_to([3.0, 2.3, 0])
        s_val = m.Text(
            "y = 0.78", font="monospace", font_size=28, color=SOFT_MAGENTA
        ).next_to(s_bar, m.UP, buff=0.2)
        s_sub = m.Text("weighted", font_size=24, color=m.GREY).next_to(
            s_bar, m.DOWN, buff=0.4
        )
        s_grad = m.MathTex(r"\nabla", color=SOFT_MAGENTA, font_size=44).next_to(
            s_sub, m.DOWN, buff=0.35
        )

        divider = m.DashedLine(
            [0, -3.0, 0], [0, 2.0, 0], color=m.GREY, stroke_width=2
        )

        self.play(
            m.FadeIn(divider),
            m.GrowFromEdge(h_bar, m.DOWN),
            m.GrowFromEdge(s_bar, m.DOWN),
            m.FadeIn(h_title), m.FadeIn(s_title),
            m.FadeIn(h_val), m.FadeIn(s_val),
            m.FadeIn(h_sub), m.FadeIn(s_sub),
        )
        self.play(
            m.Create(m.VGroup(h_nabla, h_strike)),
            m.FadeIn(s_grad, scale=1.3),
        )

        # Backprop arrow flowing back down through the magenta (soft) branch,
        # placed to the right of the bar so it never collides with the labels.
        backprop = m.CurvedArrow(
            s_bar.get_right() + 0.4 * m.RIGHT + 1.0 * m.UP,
            s_bar.get_right() + 0.4 * m.RIGHT + 1.0 * m.DOWN,
            color=SOFT_MAGENTA,
            angle=-m.TAU / 5,
        )
        grad_caption = m.Text(
            "differentiable — gradients flow through",
            font_size=26,
            color=SOFT_MAGENTA,
        ).next_to(s_grad, m.DOWN, buff=0.35)
        self.play(m.Create(backprop), m.FadeIn(grad_caption))
