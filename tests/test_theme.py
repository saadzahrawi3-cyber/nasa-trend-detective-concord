"""Presentation integrity and accessibility regressions."""
import base64
import re
import unittest
from unittest.mock import patch

import numpy as np
import plotly.graph_objects as go

from concord_theme import CSS, arc_gauge_svg, html_table, safe_prose, sparkline_svg, basemap_figure, variable_color, variable_icon
from test_concord import load_engine


class ThemeTests(unittest.TestCase):
    def test_chart_theme_preserves_numbers_and_straight_segments(self):
        ns = load_engine()
        fig = go.Figure(go.Scatter(x=[2000, 2002, 2005], y=[1, 4, 2], mode="lines+markers"))
        ns["apply_dark_monochrome_theme"](fig)
        np.testing.assert_array_equal(fig.data[0].x, [2000, 2002, 2005])
        np.testing.assert_array_equal(fig.data[0].y, [1, 4, 2])
        self.assertEqual(fig.data[0].line.shape, "linear")
        self.assertFalse(fig.layout.xaxis.showgrid)
        self.assertIsNone(fig.layout.xaxis.title.text)
        self.assertIsNone(fig.layout.yaxis.title.text)
        self.assertFalse(fig.layout.showlegend)

    def test_html_escapes_untrusted_strings_and_keeps_explanations(self):
        payload = '<img src=x onerror="alert(1)">'
        self.assertNotIn('<img', safe_prose(payload))
        self.assertIn('<strong>Evidence</strong>', safe_prose('<strong>Evidence</strong>'))
        rendered = html_table([{"Driver": payload, "r": "0.25"}])
        self.assertIn('&lt;img', rendered)
        self.assertIn('0.25', rendered)

    def test_svg_images_are_self_contained_and_linear(self):
        markup = sparkline_svg([1, 3, 2], '<script>bad</script>')
        self.assertNotIn('<script>', markup)
        encoded = re.search(r'base64,([^" ]+)', markup).group(1)
        svg = base64.b64decode(encoded).decode()
        self.assertIn('xmlns="http://www.w3.org/2000/svg"', svg)
        self.assertIn('M8.00,104.00 L160.00,16.00 L312.00,60.00', svg)
        self.assertNotIn('http://', svg.replace('http://www.w3.org/2000/svg', ''))
        gauge = base64.b64decode(re.search(r'base64,([^" ]+)', arc_gauge_svg(40)).group(1)).decode()
        self.assertIn('40.0%', gauge)
        self.assertIn('stroke-dasharray="40.00 100"', gauge)

    def test_muted_text_and_heatmap_numbers_meet_aa(self):
        def color(token):
            value = re.search(r'--' + token + r':#([0-9A-Fa-f]{6})', CSS).group(1)
            return np.array([int(value[i:i+2], 16) for i in (0, 2, 4)])
        # Highest flat fill used by selected chips, over the lighter navy base.
        surface = .11 * 255 + .89 * color('bg-1')
        text = color('text-muted')
        def luminance(rgb):
            srgb = rgb / 255
            linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
            return np.dot(linear, [.2126, .7152, .0722])
        self.assertGreaterEqual((luminance(text) + .05) / (luminance(surface) + .05), 4.5)
        for cell in ([62, 113, 151], [155, 97, 70]):
            self.assertGreaterEqual((luminance(color('text')) + .05) / (luminance(np.array(cell)) + .05), 4.5)

    def test_map_preserves_location_and_falls_back_without_tiles(self):
        meta = dict(lat=33.57, lon=-7.59, region_name='Casablanca')
        with patch('concord_theme._tile_available', return_value=True):
            fig, online = basemap_figure(meta)
        self.assertTrue(online)
        self.assertEqual(fig.data[0].type, 'scattermap')
        self.assertEqual(fig.layout.map.zoom, 5)
        self.assertEqual(list(fig.data[0].lat), [33.57])
        self.assertEqual(list(fig.data[0].lon), [-7.59])
        with patch('concord_theme._tile_available', return_value=False):
            fallback, online = basemap_figure(meta)
        self.assertFalse(online)
        self.assertEqual(fallback.data[0].type, 'scattergeo')
        self.assertEqual(list(fallback.data[0].lat), [33.57])

    def test_variable_pictograms_share_their_series_color(self):
        for code in ('T2M', 'GWETTOP', 'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN', 'QV2M', 'WS2M'):
            markup = variable_icon(code)
            svg = base64.b64decode(re.search(r'base64,([^" ]+)', markup).group(1)).decode()
            self.assertIn(variable_color(code), svg)


if __name__ == "__main__":
    unittest.main()
