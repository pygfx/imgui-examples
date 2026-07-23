"""
Colorbar with contrast limit handles
====================================

An ImGUI colorbar widget rendered with wgpu, with draggable ``vmin``/``vmax``
handles, a gamma slider, and a right-click colormap picker.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'

import numpy as np
import wgpu
from wgpu.utils.imgui import ImguiRenderer
from cmap import Colormap
from imgui_bundle import imgui
from rendercanvas.auto import RenderCanvas, loop


class ImguiColorbar:
    LUT_HEIGHT = 256
    TEX_WIDTH = 2
    HANDLE_HEIGHT = 8
    HANDLE_OVERHANG = 3  # how far a handle extends past the bar on each side
    BAR_BORDER = 1.0  # width of the outline drawn around the bar image

    def __init__(
        self,
        device,
        imgui_renderer,
        cmap_name: str,
        vmin: float,
        vmax: float,
        cmaps: list[str],
        data_range: tuple[float, float] | None = None,
        width: int = 15,
        height: int = 200,
    ):
        self._device = device
        self._renderer = imgui_renderer
        self._width = int(width)
        self._height = int(height)
        self._vmin = float(vmin)
        self._vmax = float(vmax)
        if data_range is None:
            data_range = (vmin, vmax)
        self._data_min, self._data_max = self._validate_range(data_range)
        self.gamma = 1.0

        # When True, dragging the bar between the handles moves both handles
        # together (shifts the vmin/vmax window without changing its width).
        self._region_drag = True

        # Offset (in data units) between the grabbed value and the value under
        # the cursor, captured when a drag starts so the handle tracks the
        # cursor without jumping.
        self._grab_offset = 0.0

        self._picker_tex_ids = {
            name: self._make_picker_texture(name) for name in cmaps
        }

        # The active colormap must be one of the colormaps in the picker.
        if cmap_name not in self._picker_tex_ids:
            raise ValueError(
                f"cmap_name {cmap_name!r} is not one of the available "
                f"colormaps: {list(self._picker_tex_ids)}"
            )
        self._cmap_name = cmap_name

        self._bar_texture = device.create_texture(
            size=(self.TEX_WIDTH, self.LUT_HEIGHT, 1),
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            mip_level_count=1,
            sample_count=1,
        )
        self._bar_tex_id = imgui_renderer.backend.register_texture(
            self._bar_texture.create_view()
        )
        self._update_bar_texture()

    @property
    def cmap_name(self):
        return self._cmap_name

    @cmap_name.setter
    def cmap_name(self, name):
        if name not in self._picker_tex_ids:
            raise ValueError(
                f"cmap_name {name!r} is not one of the available colormaps: "
                f"{list(self._picker_tex_ids)}"
            )
        if name != self._cmap_name:
            self._cmap_name = name
            self._update_bar_texture()

    @property
    def vmin(self):
        return self._vmin

    @vmin.setter
    def vmin(self, value):
        value = float(value)
        if value != self._vmin:
            self._vmin = value
            self._update_bar_texture()

    @property
    def vmax(self):
        return self._vmax

    @vmax.setter
    def vmax(self, value):
        value = float(value)
        if value != self._vmax:
            self._vmax = value
            self._update_bar_texture()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = int(value)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = int(value)

    @property
    def region_drag(self):
        return self._region_drag

    @region_drag.setter
    def region_drag(self, value):
        self._region_drag = bool(value)

    @property
    def data_range(self):
        return (self._data_min, self._data_max)

    @data_range.setter
    def data_range(self, value):
        self._data_min, self._data_max = self._validate_range(value)
        self._update_bar_texture()

    @staticmethod
    def _validate_range(data_range):
        data_min, data_max = float(data_range[0]), float(data_range[1])
        if data_max <= data_min:
            raise ValueError(
                f"data_range max ({data_max}) must be greater than "
                f"min ({data_min})"
            )
        return data_min, data_max

    def _make_picker_texture(self, name):
        lut = (Colormap(name)(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
        data = np.ascontiguousarray(np.tile(lut[None, :, :], (2, 1, 1)))
        h, w = data.shape[:2]
        texture = self._device.create_texture(
            size=(w, h, 1),
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            mip_level_count=1,
            sample_count=1,
        )
        self._device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            data,
            {"offset": 0, "bytes_per_row": w * 4},
            (w, h, 1),
        )
        return self._renderer.backend.register_texture(texture.create_view())

    def _update_bar_texture(self):
        span = self._data_max - self._data_min
        lo = (self.vmin - self._data_min) / span
        hi = (self.vmax - self._data_min) / span
        t = np.linspace(1.0, 0.0, self.LUT_HEIGHT)
        norm = np.clip((t - lo) / (hi - lo), 0.0, 1.0)
        colors = (Colormap(self.cmap_name)(norm) * 255).astype(np.uint8)
        data = np.ascontiguousarray(
            np.tile(colors[:, None, :], (1, self.TEX_WIDTH, 1))
        )
        self._device.queue.write_texture(
            {"texture": self._bar_texture, "mip_level": 0, "origin": (0, 0, 0)},
            data,
            {"offset": 0, "bytes_per_row": self.TEX_WIDTH * 4},
            (self.TEX_WIDTH, self.LUT_HEIGHT, 1),
        )

    def draw(self):
        imgui.text(f"{self.vmax:.4g}")

        imgui.push_style_color(imgui.Col_.border, (1.0, 1.0, 1.0, 1.0))
        imgui.push_style_var(imgui.StyleVar_.image_border_size, self.BAR_BORDER)
        imgui.image(self._bar_tex_id, image_size=(self._width, self._height))
        imgui.pop_style_var()
        imgui.pop_style_color()

        # The border insets the colored bar from the item rect, so offset by the
        # border to get the bar's content origin. This keeps the handles centered
        # on the bar rather than on the (larger) bordered image.
        item_min = imgui.get_item_rect_min()
        bar_x = item_min.x + self.BAR_BORDER
        bar_y = item_min.y + self.BAR_BORDER

        if imgui.begin_popup_context_item("##colorbar_popup"):
            self._draw_popup()
            imgui.end_popup()

        imgui.text(f"{self.vmin:.4g}")
        self._draw_handles(bar_x, bar_y, self._width, self._height)

    def _value_to_y(self, v, y0, bar_h):
        span = self._data_max - self._data_min
        return y0 + (1.0 - (v - self._data_min) / span) * bar_h

    def _y_to_value(self, y, y0, bar_h):
        span = self._data_max - self._data_min
        return self._data_min + (1.0 - (y - y0) / bar_h) * span

    def _draw_handles(self, bar_x, bar_y, bar_w, bar_h):
        draw_list = imgui.get_window_draw_list()
        fill = imgui.color_convert_float4_to_u32((1.0, 1.0, 1.0, 1.0))
        outline = imgui.color_convert_float4_to_u32((0.0, 0.0, 0.0, 1.0))
        h = self.HANDLE_HEIGHT
        # Extend the handle HANDLE_OVERHANG pixels past the bar's outer (bordered)
        # edge on each side.
        x0 = bar_x - self.BAR_BORDER - self.HANDLE_OVERHANG
        x1 = bar_x + bar_w + self.BAR_BORDER + self.HANDLE_OVERHANG
        span = self._data_max - self._data_min
        min_sep = (h / bar_h) * span

        def cursor_value():
            # The data value under the cursor. Handles track this absolute
            # position (plus the grab offset) rather than accumulating per-frame
            # deltas, so a fast drag past the top/bottom edge pins the handle to
            # the extreme instead of stalling partway.
            return self._y_to_value(imgui.get_io().mouse_pos.y, bar_y, bar_h)

        # Drag the region between the handles to move both handles together.
        if self.region_drag:
            top = self._value_to_y(self.vmax, bar_y, bar_h) + h / 2
            bottom = self._value_to_y(self.vmin, bar_y, bar_h) - h / 2
            if bottom > top:
                imgui.set_cursor_screen_pos((x0, top))
                imgui.invisible_button("##colorbar_region", (x1 - x0, bottom - top))
                if imgui.is_item_activated():
                    self._grab_offset = 0.5 * (self.vmin + self.vmax) - cursor_value()
                if imgui.is_item_active():
                    half = 0.5 * (self.vmax - self.vmin)
                    center = cursor_value() + self._grab_offset
                    center = max(self._data_min + half,
                                 min(self._data_max - half, center))
                    vmin, vmax = center - half, center + half
                    if (vmin, vmax) != (self._vmin, self._vmax):
                        self._vmin, self._vmax = vmin, vmax
                        self._update_bar_texture()

        for v in (self.vmax, self.vmin):
            y = self._value_to_y(v, bar_y, bar_h)
            draw_list.add_rect_filled((x0, y - h / 2), (x1, y + h / 2), fill)
            draw_list.add_rect(
                (x0, y - h / 2), (x1, y + h / 2), outline, thickness=1.0
            )

        for label, attr, lo_fn, hi_fn in (
            ("##vmax_handle", "vmax",
             lambda: self.vmin + min_sep, lambda: self._data_max),
            ("##vmin_handle", "vmin",
             lambda: self._data_min, lambda: self.vmax - min_sep),
        ):
            cur = getattr(self, attr)
            y = self._value_to_y(cur, bar_y, bar_h)
            imgui.set_cursor_screen_pos((x0, y - h / 2))
            imgui.invisible_button(label, (x1 - x0, h))
            if imgui.is_item_activated():
                self._grab_offset = cur - cursor_value()
            if imgui.is_item_active():
                setattr(
                    self, attr,
                    max(lo_fn(), min(hi_fn(), cursor_value() + self._grab_offset)),
                )

    def _draw_popup(self):
        imgui.set_next_item_width(150)
        _, self.gamma = imgui.slider_float("gamma", self.gamma, 0.1, 5.0)
        imgui.separator()
        texture_height = imgui.get_font_size() - 2
        for name, tex_id in self._picker_tex_ids.items():
            # menu_item's own checkmark is drawn at the far right of the row, so
            # pass p_selected=False and draw our own checkmark next to the image.
            if imgui.menu_item(name, "", False)[0]:
                self.cmap_name = name
            imgui.same_line()
            imgui.push_style_color(imgui.Col_.border, (1.0, 1.0, 1.0, 1.0))
            imgui.push_style_var(imgui.StyleVar_.image_border_size, 1.0)
            imgui.image(tex_id, image_size=(100, texture_height))
            imgui.pop_style_var()
            imgui.pop_style_color()
            if self.cmap_name == name:
                imgui.same_line(0.0, 4.0)
                pos = imgui.get_cursor_screen_pos()
                imgui.internal.render_check_mark(
                    imgui.get_window_draw_list(),
                    (pos.x, pos.y),
                    imgui.get_color_u32(imgui.Col_.text),
                    texture_height,
                )
                imgui.dummy((texture_height, texture_height))


canvas = RenderCanvas(size=(256, 400), max_fps=60, update_mode="continuous")
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()
imgui_renderer = ImguiRenderer(device, canvas)

colorbar = ImguiColorbar(
    device=device,
    imgui_renderer=imgui_renderer,
    cmap_name="green",
    vmin=0.0,
    vmax=1.0,
    cmaps=["green", "red", "blue", "gray"],
)


def update_gui():
    imgui.set_next_window_pos((20, 20), imgui.Cond_.appearing)
    # Give the window an explicit initial size so it renders on the first frame.
    # An auto-fit window has no size until its second frame, which would leave
    # the gallery screenshot (a single rendered frame) blank.
    imgui.set_next_window_size((120, 340), imgui.Cond_.appearing)
    imgui.begin("colorbar", None)
    colorbar.draw()
    imgui.end()


imgui_renderer.set_gui(update_gui)
canvas.request_draw(imgui_renderer.render)


if __name__ == "__main__":
    loop.run()
