import hashlib
import io
import random
from collections.abc import Callable
from functools import partial, wraps
from typing import ParamSpec

import equinox as eqx
import jax
import jax.numpy as jnp
import manim as m
from manim_slides import Slide, ThreeDSlide
import numpy as np
import softjax as sj
from jaxtyping import Array, Bool, Int
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


class Presentation(ThreeDSlide):
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
        #self.heaviside_and_bools()
        #self.next_slide()
        #self.comparisons_and_logic()
        #self.next_slide()
        #self.axiswise()
        #self.next_slide()
        #self.softsort()
        #self.next_slide()
        #self.library_overview()
        #self.next_slide()
        #self.sorting_benchmark()
        #self.next_slide()
        self.straight_through()
        self.next_slide()

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

    def axiswise(self):
        """Indexing as an axiswise operator: standard jax vs. SoftJAX hard/soft."""
        self.new_clean_slide("Axiswise operators")
        accent = SOFT_MODES[0][1]

        def code_box(code_string):
            return m.Code(
                code_string=code_string,
                language="python",
                add_line_numbers=False,
                formatter_style="default",
                background_config=dict(fill_color=CODE_BG_COLOR, stroke_width=0),
            ).scale(0.7)

        # Comments removed from the code; the result is shown to the right
        # of each box via an arrow + illustration instead.
        sections = [
            (
                "Standard indexing",
                "x = jnp.array([0.1, 0.4, 0.8])\n"
                "idx = jnp.argmax(x)\n"
                "y = x[idx]",
            ),
            (
                'SoftJax, mode="hard"',
                'hard_idx = sj.argmax(x, mode="hard")\n'
                "y = sj.dynamic_index_in_dim(x, hard_idx)",
            ),
            (
                'SoftJax, mode="soft"',
                "soft_idx = sj.argmax(x)\n"
                "y = sj.dynamic_index_in_dim(x, soft_idx)",
            ),
        ]

        blocks = []
        for label_text, code_string in sections:
            label = m.Text(label_text, font_size=28, color=m.BLACK)
            code = code_box(code_string)
            block = m.VGroup(label, code).arrange(
                m.DOWN, buff=0.2, aligned_edge=m.LEFT
            )
            blocks.append(block)

        column = (
            m.VGroup(*blocks)
            .arrange(m.DOWN, buff=0.85, aligned_edge=m.LEFT)
            .to_edge(m.LEFT, buff=0.9)
            .to_edge(m.UP, buff=1.6)
        )

        def result_arrow(code):
            tail = code.get_right() + 0.2 * m.RIGHT
            return m.Arrow(
                tail, tail + 1.1 * m.RIGHT, buff=0.0,
                color=m.BLACK, stroke_width=4,
                max_tip_length_to_length_ratio=0.25,
            )

        # ---- Section 1 illustration: plain index / value readout ----
        arrow1 = result_arrow(blocks[0][1])
        result1 = m.Text(
            "Index: 2, Value: 0.8", font_size=28, color=m.BLACK
        ).next_to(arrow1, m.RIGHT, buff=0.3)

        # ---- Section 2 illustration: one-hot bar plot . column vector = 0.8 ----
        def one_hot_bars(values):
            unit, bw = 0.7, 0.3
            bars = m.VGroup(
                *[
                    m.Rectangle(
                        width=bw, height=max(v * unit, 0.02),
                        fill_color=accent, fill_opacity=1.0,
                        stroke_width=1.0, stroke_color=m.BLACK,
                    )
                    for v in values
                ]
            ).arrange(m.RIGHT, buff=0.34, aligned_edge=m.DOWN)
            baseline = m.Line(
                bars.get_corner(m.DL) + 0.06 * m.LEFT,
                bars.get_corner(m.DR) + 0.06 * m.RIGHT,
                color=m.GREY, stroke_width=2,
            )
            labels = m.VGroup(
                *[
                    m.MathTex(str(v), font_size=20, color=m.BLACK).next_to(
                        bars[i], m.DOWN, buff=0.16
                    )
                    for i, v in enumerate(values)
                ]
            )
            return m.VGroup(bars, baseline, labels)

        bars = one_hot_bars([0, 0, 1])
        dot = m.MathTex(r"\cdot", font_size=72, color=m.BLACK)
        colvec = m.Matrix(
            [["0.1"], ["0.4"], ["0.8"]], bracket_h_buff=0.12
        ).scale(0.55).set_color(m.BLACK)
        eq = m.MathTex("=", font_size=56, color=m.BLACK)
        val = m.MathTex("0.8", font_size=44, color=m.BLACK)
        illus2 = m.VGroup(bars, dot, colvec, eq, val).arrange(m.RIGHT, buff=0.32)
        arrow2 = result_arrow(blocks[1][1])
        illus2.next_to(arrow2, m.RIGHT, buff=0.3)

        # Coding-font captions naming the two operands.
        bars_label = m.Text(
            "hard_idx", font="Monospace", font_size=22, color=m.BLACK
        ).next_to(bars, m.UP, buff=0.25)
        colvec_label = m.Text(
            "array", font="Monospace", font_size=22, color=m.BLACK
        ).next_to(colvec, m.UP, buff=0.25)

        # Small sketchy arrow + caption pointing at the selected value.
        mean_text = m.Text("Mean value", font_size=22, color=m.BLACK).next_to(
            val, m.UP, buff=0.7
        )
        sketch = m.CurvedArrow(
            mean_text.get_bottom() + 0.05 * m.DOWN,
            val.get_top() + 0.06 * m.UP,
            angle=-m.PI / 3, color=m.BLACK, stroke_width=2.5, tip_length=0.16,
        )

        # ---- Section 3 illustration: soft probabilities . column vector = 0.78 ----
        bars3 = one_hot_bars([0.004, 0.042, 0.953])
        dot3 = m.MathTex(r"\cdot", font_size=72, color=m.BLACK)
        colvec3 = m.Matrix(
            [["0.1"], ["0.4"], ["0.8"]], bracket_h_buff=0.12
        ).scale(0.55).set_color(m.BLACK)
        eq3 = m.MathTex("=", font_size=56, color=m.BLACK)
        val3 = m.MathTex("0.78", font_size=44, color=m.BLACK)
        illus3 = m.VGroup(bars3, dot3, colvec3, eq3, val3).arrange(m.RIGHT, buff=0.32)
        arrow3 = result_arrow(blocks[2][1])
        illus3.next_to(arrow3, m.RIGHT, buff=0.3)

        bars_label3 = m.Text(
            "soft_idx", font="Monospace", font_size=22, color=m.BLACK
        ).next_to(bars3, m.UP, buff=0.25)
        colvec_label3 = m.Text(
            "array", font="Monospace", font_size=22, color=m.BLACK
        ).next_to(colvec3, m.UP, buff=0.25)
        mean_text3 = m.Text(
            "Mean value", font_size=22, color=m.BLACK
        ).next_to(val3, m.UP, buff=0.7)
        sketch3 = m.CurvedArrow(
            mean_text3.get_bottom() + 0.05 * m.DOWN,
            val3.get_top() + 0.06 * m.UP,
            angle=-m.PI / 3, color=m.BLACK, stroke_width=2.5, tip_length=0.16,
        )

        content = m.VGroup(
            column, arrow1, result1, arrow2, illus2,
            bars_label, colvec_label, mean_text, sketch,
            arrow3, illus3, bars_label3, colvec_label3, mean_text3, sketch3,
        )
        # Defensive: shrink + recenter if the whole layout overflows the frame.
        max_h = m.config.frame_height - 1.3
        max_w = m.config.frame_width - 1.0
        scale = min(max_h / content.height, max_w / content.width, 1.0)
        if scale < 1.0:
            content.scale(scale)
        content.to_edge(m.LEFT, buff=0.9).to_edge(m.UP, buff=1.4)

        # ---- Beat 1: standard indexing, then its index/value readout ----
        label0, code0 = blocks[0]
        self.play(m.FadeIn(label0, shift=0.2 * m.DOWN))
        self.play(m.FadeIn(code0, shift=0.2 * m.DOWN))
        self.next_slide()
        self.play(m.GrowArrow(arrow1), m.FadeIn(result1, shift=0.2 * m.RIGHT))
        self.next_slide()

        # ---- Beat 2: hard mode, then one-hot . column-vector = value ----
        label1, code1 = blocks[1]
        self.play(m.FadeIn(label1, shift=0.2 * m.DOWN))
        self.play(m.FadeIn(code1, shift=0.2 * m.DOWN))
        self.next_slide()
        self.play(
            m.GrowArrow(arrow2),
            m.FadeIn(illus2, shift=0.2 * m.RIGHT),
            m.FadeIn(bars_label, shift=0.1 * m.DOWN),
            m.FadeIn(colvec_label, shift=0.1 * m.DOWN),
        )
        self.play(m.FadeIn(mean_text, shift=0.1 * m.DOWN), m.Create(sketch))
        self.next_slide()

        # ---- Beat 3: soft mode, then soft-probabilities . array = value ----
        label2, code2 = blocks[2]
        self.play(m.FadeIn(label2, shift=0.2 * m.DOWN))
        self.play(m.FadeIn(code2, shift=0.2 * m.DOWN))
        self.next_slide()
        self.play(
            m.GrowArrow(arrow3),
            m.FadeIn(illus3, shift=0.2 * m.RIGHT),
            m.FadeIn(bars_label3, shift=0.1 * m.DOWN),
            m.FadeIn(colvec_label3, shift=0.1 * m.DOWN),
        )
        self.play(m.FadeIn(mean_text3, shift=0.1 * m.DOWN), m.Create(sketch3))

    def softsort(self):
        """Example: SoftSort — walk through the build-up images in order."""
        self.new_clean_slide("Example: SoftSort")

        image_files = [
            "images/softsort/softsort_slide1.png",
            "images/softsort/softsort_slide2.png",
            "images/softsort/softsort_slide3.png",
            "images/softsort/softsort_slide4.png",
        ]
        # Each image is a full 16:9 build step; fit it below the slide title
        # and let successive steps replace one another in ascending order.
        target_h = m.config.frame_height - 1.2
        prev = None
        for i, path in enumerate(image_files):
            img = m.ImageMobject(path)
            img.height = target_h
            img.to_edge(m.DOWN, buff=0.15)
            if prev is None:
                self.play(m.FadeIn(img))
            else:
                self.play(m.FadeOut(prev), m.FadeIn(img))
            prev = img
            if i < len(image_files) - 1:
                self.next_slide()

    def library_overview(self):
        """Library overview — booktabs-style table of all SoftJAX operators."""
        self.new_clean_slide("Library overview")

        header_fs = 26
        cell_fs = 22

        def mono(text, fs=cell_fs):
            return m.Text(text, font="Monospace", font_size=fs, color=m.BLACK)

        def header(text):
            return m.Text(text, font_size=header_fs, color=m.BLACK, weight=m.BOLD)

        def checkmark():
            return m.VGroup(
                m.Line([-0.5, 0.0, 0], [-0.18, -0.42, 0]),
                m.Line([-0.18, -0.42, 0], [0.5, 0.5, 0]),
            ).set_stroke(color=m.GREEN_D, width=4).scale(0.16)

        def check(text, mono_body):
            label = mono(text) if mono_body else m.Text(
                text, font_size=cell_fs, color=m.BLACK)
            return m.VGroup(checkmark(), label).arrange(m.RIGHT, buff=0.16)

        # --- The four operator columns ---
        columns = [
            ("Axiswise", ["argmin", "min", "argsort", "sort", "argquantile",
                          "quantile", "argmedian", "median", "top_k", "rank"]),
            ("Elementwise", ["heaviside", "round", "sign", "abs", "relu", "clip",
                             "less", "less_equal", "equal", "not_equal", "isclose"]),
            ("Logical", ["logical_and", "logical_or", "logical_xor",
                         "logical_not", "any", "all"]),
            ("Selection", ["where", "take_along_axis", "take", "choose",
                           "dynamic_index_in_dim", "dynamic_slice_in_dim",
                           "dynamic_slice"]),
        ]
        col_groups = []
        for head, items in columns:
            col = m.VGroup(header(head), *[mono(it) for it in items])
            col.arrange(m.DOWN, buff=0.20, aligned_edge=m.LEFT)
            col_groups.append(col)

        # --- The Modes / Methods column ---
        modes_col = m.VGroup(
            header("Modes"),
            *[check(md, True) for md in ["smooth", "c0", "c1", "c2"]],
            header("Methods"),
            *[check(mt, False) for mt in ["Optimal Transport", "SoftSort",
                                          "NeuralSort", "FastSoftSort",
                                          "SmoothSort", "Sorting Network"]],
        ).arrange(m.DOWN, buff=0.20, aligned_edge=m.LEFT)
        # Breathing room above the "Methods" sub-header.
        modes_col[5].shift(0.12 * m.DOWN)
        for cell in modes_col[6:]:
            cell.shift(0.12 * m.DOWN)

        all_cols = col_groups + [modes_col]
        table = m.VGroup(*all_cols).arrange(m.RIGHT, buff=0.6, aligned_edge=m.UP)

        # --- Booktabs rules ---
        left = table.get_left()[0] - 0.12
        right = table.get_right()[0] + 0.12
        top_y = table.get_top()[1] + 0.14
        mid_y = min(c[0].get_bottom()[1] for c in all_cols) - 0.07
        bot_y = table.get_bottom()[1] - 0.14

        def hline(y, w):
            return m.Line([left, y, 0], [right, y, 0], color=m.BLACK, stroke_width=w)

        rules = m.VGroup(hline(top_y, 3.0), hline(mid_y, 1.5), hline(bot_y, 3.0))

        # Keep the table compact and centred so there is a band of whitespace
        # above (for the Modes annotation) and below (for the Elementwise one).
        content = m.VGroup(table, rules)
        max_h = 4.0
        max_w = m.config.frame_width - 0.9
        scale = min(max_h / content.height, max_w / content.width, 1.0)
        if scale < 1.0:
            content.scale(scale)
        content.move_to([0, 0.45, 0])

        self.play(
            m.Create(rules),
            m.LaggedStart(*[m.FadeIn(c[0], shift=0.15 * m.DOWN) for c in all_cols],
                          lag_ratio=0.15),
        )
        self.play(
            m.LaggedStart(*[m.FadeIn(m.VGroup(*c[1:]), shift=0.15 * m.UP)
                            for c in all_cols], lag_ratio=0.12),
        )
        self.next_slide()

        # --- Annotation 1: arrow from a note down to the Modes column ---
        modes_note = m.Text(
            "Appendix: Proof of smoothness of p-norm regularized projections.",
            font_size=18, color=m.BLACK, line_spacing=0.85,
        ).to_corner(m.UR, buff=0.5)
        m_end = modes_col[0].get_top() + 0.14 * m.UP
        modes_arrow = m.Arrow(
            m_end + 0.6 * m.UP, m_end,
            color=m.BLACK, buff=0.0, stroke_width=3,
            max_tip_length_to_length_ratio=0.35,
        )
        self.play(m.FadeIn(modes_note, shift=0.15 * m.DOWN))
        self.play(m.GrowArrow(modes_arrow))
        self.next_slide()

        # --- Annotation 2: arrow from a note up into the Elementwise column ---
        elem_note = m.Text(
            "Appendix: Elementwise operators are special case of axiswise operators.",
            font_size=18, color=m.BLACK, line_spacing=0.85,
        )
        elem_col = col_groups[1]
        elem_note.next_to(elem_col, m.DOWN, buff=1.05)
        elem_note.set_x(elem_col.get_x())
        e_end = elem_col.get_bottom() + 0.14 * m.DOWN
        elem_arrow = m.Arrow(
            e_end + 0.6 * m.DOWN, e_end,
            color=m.BLACK, buff=0.0, stroke_width=3,
            max_tip_length_to_length_ratio=0.35,
        )
        self.play(m.FadeIn(elem_note, shift=0.15 * m.UP))
        self.play(m.GrowArrow(elem_arrow))

    def sorting_benchmark(self):
        """Untitled slide: bold 'Sorting' heading above the sort benchmark."""
        self.new_clean_slide("")

        heading = m.Text("Sorting", weight=m.BOLD, font_size=24, color=m.BLACK)
        plot = m.ImageMobject("images/benchmark_sort_smooth.png")

        group = m.Group(heading, plot).arrange(m.DOWN, buff=0.2)
        max_h = m.config.frame_height - 0.5
        max_w = m.config.frame_width - 0.5
        # Scale up (or down) to fill the frame as much as possible.
        group.scale(min(max_h / group.height, max_w / group.width))
        group.move_to(0.35 * m.DOWN)

        self.play(m.FadeIn(heading, shift=0.2 * m.DOWN))
        self.play(m.FadeIn(plot, shift=0.2 * m.UP))

    def straight_through(self):
        """Straight-through estimation: the trick (left) + a plot (right). The
        2D (front) view shows relu(x) for hard and smooth modes as line plots in
        the x-z plane; it then fades out as the 3D surface f(x,y) = y*relu_st(x)
        and its normalized jax.grad field fade in and the camera pans to an
        isometric view."""
        self.new_clean_slide("STE")

        # Start looking along the y-axis (x to the right, z up) so the surface
        # reads like a flat 2D plot of relu(x). The slide title / number live in
        # the canvas; pin them (and the left-hand text) to the frame so the
        # camera pan never tilts them.
        self.set_camera_orientation(phi=90 * m.DEGREES, theta=-90 * m.DEGREES)
        self.add_fixed_in_frame_mobjects(self.slide_title, self.slide_number)

        # ---------------- Upper-left: the STE trick box ------------------
        formula = m.MathTex(
            r"f_{\text{STE}}(x) = \operatorname{sg}(f(x)) + f_{\tau}(x)"
            r" - \operatorname{sg}(f_{\tau}(x))",
            font_size=28,
        )
        formula_box = m.SurroundingRectangle(
            formula, buff=0.3, corner_radius=0.15,
            color=SOFT_MODES[0][1], stroke_width=2.5,
        )
        boxed = m.VGroup(formula, formula_box)
        if boxed.width > 5.3:
            boxed.scale(5.3 / boxed.width)
        # Caption tag on the box's upper-left corner (not bold).
        trick_tag = m.Text(
            "Straight-through trick", font_size=22, color=m.BLACK
        )
        trick_tag.next_to(boxed, m.UP, buff=0.1).align_to(boxed, m.LEFT)
        trick = m.VGroup(trick_tag, boxed)
        trick.to_corner(m.UL, buff=0.4).shift(1.0 * m.DOWN)

        # ---------------- Below it: the STE pitfall box ------------------
        pitfall_formula = m.MathTex(
            r"\left(f \cdot g\right)_{\mathrm{STE}}"
            r" \neq f_{\mathrm{STE}} \cdot g_{\mathrm{STE}}",
            font_size=28,
        )
        pitfall_box = m.SurroundingRectangle(
            pitfall_formula, buff=0.3, corner_radius=0.15,
            color=INVALID_COLOR, stroke_width=2.5,
        )
        pitfall_boxed = m.VGroup(pitfall_formula, pitfall_box)
        if pitfall_boxed.width > 5.3:
            pitfall_boxed.scale(5.3 / pitfall_boxed.width)
        pitfall_tag = m.Text(
            "Straight-through pitfall", font_size=22, color=m.BLACK
        )
        pitfall_tag.next_to(pitfall_boxed, m.UP, buff=0.1).align_to(
            pitfall_boxed, m.LEFT
        )
        pitfall_note = m.VGroup(pitfall_tag, pitfall_boxed)
        pitfall_note.next_to(trick, m.DOWN, buff=0.7, aligned_edge=m.LEFT)

        # 2D-view legend (hard vs smooth relu) and the 3D-view label + the
        # gradient-arrow legend. All pinned to the frame.
        def swatch(color):
            return m.Line(m.ORIGIN, 0.5 * m.RIGHT, color=color, stroke_width=4)

        legend_2d = m.VGroup(
            m.VGroup(
                swatch(HARD_COLOR),
                m.Text('sj.relu(x, mode="hard")', font="Monospace",
                       font_size=20, color=m.BLACK),
            ).arrange(m.RIGHT, buff=0.15),
            m.VGroup(
                swatch(SOFT_MODES[0][1]),
                m.Text('sj.relu(x, mode="smooth")', font="Monospace",
                       font_size=20, color=m.BLACK),
            ).arrange(m.RIGHT, buff=0.15),
        ).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT).move_to([3.0, 2.2, 0])

        label_relu_st = m.Text(
            'sj.st(sj.relu)(x)',
            font="Monospace", font_size=24, color=m.BLACK,
        ).move_to([2.9, 2.3, 0])
        label_saddle = m.Text(
            'y * sj.st(sj.relu)(x)',
            font="Monospace", font_size=24, color=m.BLACK,
        ).move_to([2.9, 2.3, 0])
        # The last beat's title is a code box (replaces the text title) showing
        # the straight-through of the whole product.
        code_text = m.Paragraph(
            "@sj.st",
            "def f(x, y):",
            "    return y * sj.relu(x)",
            font="Monospace", font_size=20, color=m.BLACK, line_spacing=0.6,
        )
        code_bg = m.SurroundingRectangle(
            code_text, buff=0.22, corner_radius=0.1,
            color=m.GREY, stroke_width=0.0,
            fill_color=CODE_BG_COLOR, fill_opacity=1.0,
        )
        code_box = m.VGroup(code_bg, code_text).move_to([3.0, 3.0, 0])
        # Legend describing the in-plane gradient arrows: to the right of the
        # plot, slightly below the x-axis.
        grad_legend = m.VGroup(
            m.Arrow(
                m.ORIGIN, 0.55 * m.RIGHT, color=m.BLACK, buff=0.0,
                stroke_width=3, max_tip_length_to_length_ratio=0.4,
            ),
            m.Text("gradients", font="Monospace", font_size=22, color=m.BLACK),
        ).arrange(m.RIGHT, buff=0.18).move_to([5.0, -0.7, 0])

        self.add_fixed_in_frame_mobjects(
            trick, pitfall_note, legend_2d, label_relu_st, label_saddle,
            code_box, grad_legend,
        )
        self.remove(
            trick, pitfall_note, legend_2d, label_relu_st, label_saddle,
            code_box, grad_legend,
        )

        # ---------------------- Right: the 3D surface --------------------
        axes = m.ThreeDAxes(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1], z_range=[-4, 4, 2],
            x_length=4.5, y_length=4.5, z_length=3.0,
            axis_config=dict(color=m.GREY_D, stroke_width=2),
            tips=False,
        ).shift(2.6 * m.RIGHT + 0.5 * m.UP)

        # Faint reference grid on the z = 0 floor, to read the 3D surface
        # against (revealed with the isometric/3D view).
        floor_grid = m.VGroup()
        for gx in np.linspace(-2, 2, 9):
            floor_grid.add(m.Line(axes.c2p(gx, -2, 0), axes.c2p(gx, 2, 0)))
        for gy in np.linspace(-2, 2, 9):
            floor_grid.add(m.Line(axes.c2p(-2, gy, 0), axes.c2p(2, gy, 0)))
        floor_grid.set_stroke(color=m.GREY_B, width=1, opacity=0.5)

        # 2D-view content: relu(x) for hard and smooth modes as line plots in
        # the x-z plane (y = 0), evaluated directly through softjax.
        def relu_line(mode, color, softness=None):
            xs = np.linspace(-2, 2, 200)
            if softness is None:
                zs = np.asarray(sj.relu(jnp.asarray(xs), mode=mode))
            else:
                zs = np.asarray(sj.relu(jnp.asarray(xs), mode=mode, softness=softness))
            pts = [axes.c2p(float(x), 0.0, float(z)) for x, z in zip(xs, zs)]
            return (
                m.VMobject()
                .set_points_as_corners(pts)
                .set_stroke(color=color, width=6)
            )

        relu_smooth_line = relu_line("smooth", SOFT_MODES[0][1], softness=0.5)
        relu_hard_line = relu_line("hard", HARD_COLOR)
        lines_2d = m.VGroup(relu_smooth_line, relu_hard_line)

        # 3D-view content. First surface: f = relu_st(x) (a ramp, constant in
        # y). Second surface: f = y * relu_st(x). Each carries its own
        # normalized jax.grad field, drawn as black in-plane arrows.
        def relu_st_z(u, v):
            return float(
                sj.relu_st(jnp.asarray(u, dtype=jnp.float32))
            )

        def saddle_z(u, v):
            return float(v) * relu_st_z(u, v)

        def make_surface(zfun, zmax):
            surf = m.Surface(
                lambda u, v: axes.c2p(u, v, zfun(u, v)),
                u_range=[-2, 2], v_range=[-2, 2],
                resolution=(24, 24),
                fill_opacity=0.95, stroke_width=0.8, stroke_color=m.GREY,
            )
            # coolwarm: negative -> red, zero -> grey, positive -> blue.
            surf.set_fill_by_value(
                axes=axes,
                colorscale=[(m.RED, -zmax), (m.GREY_A, 0.0), (m.BLUE, zmax)],
                axis=2,
            )
            return surf

        # Last beat's function: sj.st applied to the *whole* product
        # y * relu(x). Its smooth backward (softness 0.5) yields informative
        # gradients everywhere -- unlike y * sj.st(relu)(x), whose field
        # vanishes for x < 0 -- illustrating (f*g)_STE != f_STE * g_STE.
        @sj.st
        def st_product(x, y, mode="smooth", softness=0.5):
            return y * sj.relu(x, mode=mode, softness=softness)

        relu_st_grad = jax.grad(lambda p: sj.relu_st(p[0]))
        saddle_grad = jax.grad(lambda p: p[1] * sj.relu_st(p[0]))
        product_grad = jax.grad(lambda p: st_product(p[0], p[1]))

        def make_grad_field(grad_fn, arrow_len=0.45):
            arrows = m.VGroup()
            for x in np.linspace(-1.5, 1.5, 7):
                for y in np.linspace(-1.5, 1.5, 5):
                    g = np.asarray(grad_fn(jnp.asarray([x, y], dtype=jnp.float32)))
                    mag = float((g[0] ** 2 + g[1] ** 2) ** 0.5)
                    if mag < 1e-10:
                        continue
                    ux, uy = float(g[0]) / mag, float(g[1]) / mag
                    start = axes.c2p(x, y, 0.0)
                    end = axes.c2p(x + arrow_len * ux, y + arrow_len * uy, 0.0)
                    arrows.add(
                        m.Arrow(
                            start, end, color=m.BLACK, buff=0.0,
                            stroke_width=3, tip_length=0.14,
                            max_tip_length_to_length_ratio=0.4,
                        )
                    )
            return arrows

        relu_st_surface = make_surface(relu_st_z, 2.0)
        relu_st_field = make_grad_field(relu_st_grad)
        saddle_surface = make_surface(saddle_z, 2.0)
        saddle_field = make_grad_field(saddle_grad)
        # sj.st(y * relu)(x): same forward surface as the saddle, but a
        # gradient field that is informative everywhere (the correct STE).
        product_surface = make_surface(saddle_z, 2.0)
        product_field = make_grad_field(product_grad)

        # Beat 1: front (2D) view — relu hard vs smooth as line plots.
        self.play(
            m.FadeIn(trick),
            m.FadeIn(legend_2d),
            m.Create(axes),
        )
        self.play(m.Create(relu_smooth_line), m.Create(relu_hard_line))
        self.next_slide()

        # Beat 2: fade the 2D lines out and the relu_st(x) surface + gradient
        # arrows in, while panning from the front view to the isometric view.
        self.move_camera(
            phi=70 * m.DEGREES, theta=-60 * m.DEGREES, run_time=2.0,
            added_anims=[
                m.FadeOut(lines_2d),
                m.FadeOut(legend_2d),
                m.FadeIn(relu_st_surface),
                m.FadeIn(relu_st_field),
                m.FadeIn(floor_grid),
                m.FadeIn(label_relu_st),
                m.FadeIn(grad_legend),
            ],
        )
        self.next_slide()

        # Beat 3: swap relu_st(x) for the full y * sj.st(relu)(x) surface and
        # its gradient field, and raise the straight-through pitfall.
        self.play(
            m.ReplacementTransform(relu_st_surface, saddle_surface),
            m.ReplacementTransform(relu_st_field, saddle_field),
            m.FadeOut(label_relu_st),
            m.FadeIn(label_saddle),
            m.FadeIn(pitfall_note, shift=0.15 * m.DOWN),
        )
        self.next_slide()

        # Beat 4: swap to sj.st(y * relu)(x) -- same surface, but its gradient
        # field is informative everywhere (the correct STE for the product).
        self.play(
            m.ReplacementTransform(saddle_surface, product_surface),
            m.ReplacementTransform(saddle_field, product_field),
            m.FadeOut(label_saddle),
            m.FadeIn(code_box),
        )
