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
        self.intro()
        self.next_slide()
        self.heaviside_and_bools()
        #self.next_slide()
        #self.comparisons_and_logic()
        #self.next_slide()
        #self.relu()
        #self.next_slide()
        #self.argmax()
        #self.next_slide()
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
            "Authors: Anselm Paulus*  A. René Geist*  Vít Musil  Sebastian Hoffmann  Georg Martius",
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

    def heaviside_and_bools(self):
        pass
    