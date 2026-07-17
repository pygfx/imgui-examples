"""Sphinx configuration for the imgui-examples gallery."""

import os
from pathlib import Path

# Force offscreen rendering before wgpu/rendercanvas pick a backend, so the
# gallery scraper can grab a frame without opening a window. This must happen
# before pygfx (and therefore wgpu) is imported below.
os.environ["RENDERCANVAS_FORCE_OFFSCREEN"] = "1"

from pygfx.utils.gallery_scraper import find_examples_for_gallery

ROOT_DIR = Path(__file__).parents[1]
EXAMPLES_DIR = ROOT_DIR / "examples"

# -- Project information -----------------------------------------------------

project = "imgui-examples"
copyright = "2026, imgui-examples contributors"
author = "imgui-examples contributors"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx_gallery.gen_gallery",
]

html_theme = "pydata_sphinx_theme"

# The scraper tuple is not picklable, which trips Sphinx' config cache.
# See https://github.com/sphinx-doc/sphinx/issues/12300
suppress_warnings = ["config.cache"]

# -- Sphinx-gallery ----------------------------------------------------------

sphinx_gallery_conf = {
    "gallery_dirs": "_gallery",
    # Reuse pygfx's scraper: for each executed example it finds the rendercanvas
    # in the example's namespace, calls ``canvas.draw()``, and writes the
    # resulting frame as a lossless ``.webp`` thumbnail. The imgui UI is present
    # in that frame because each example registers ``imgui_renderer.render`` as
    # the canvas draw callback at module level.
    "image_scrapers": ("pygfx",),
    # Strip the ``# sphinx_gallery_pygfx_docs`` directive from the shown source.
    "remove_config_comments": True,
    "thumbnail_size": (400, 400),
}

# Sets "examples_dirs", "filename_pattern" and "ignore_pattern" based on the
# ``# sphinx_gallery_pygfx_docs`` comment in each example.
_extra_conf = find_examples_for_gallery(EXAMPLES_DIR)

# find_examples_for_gallery emits "()" for the ignore pattern when there are no
# files to hide. That empty alternation matches every path and would exclude
# all examples, so swap it for a regex that never matches.
if _extra_conf.get("ignore_pattern") == "()":
    _extra_conf["ignore_pattern"] = r"(?!)"

sphinx_gallery_conf.update(_extra_conf)
