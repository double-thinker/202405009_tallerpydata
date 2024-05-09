from functools import partial

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Accordion, Box, HBox, Layout, VBox

from tools.stablemapping import StableMapping
from tools.tokentools import split_tokens
from tools.tokenwidget import TokenMatrix, css

tab20_pallete = [
    "rgb(31, 119, 180)",
    "rgb(174, 199, 232)",
    "rgb(255, 127, 14)",
    "rgb(255, 187, 120)",
    "rgb(44, 160, 44)",
    "rgb(152, 223, 138)",
    "rgb(214, 39, 40)",
    "rgb(255, 152, 150)",
    "rgb(148, 103, 189)",
    "rgb(197, 176, 213)",
    "rgb(140, 86, 75)",
    "rgb(196, 156, 148)",
    "rgb(227, 119, 194)",
    "rgb(247, 182, 210)",
    "rgb(127, 127, 127)",
    "rgb(199, 199, 199)",
    "rgb(188, 189, 34)",
    "rgb(219, 219, 141)",
    "rgb(23, 190, 207)",
    "rgb(158, 218, 229)",
]

colorblind_pallete = [
    "rgb(90,169,215)",
    "#009292",
    "#ffb6db",
    "#006ddb",
    "#b66dff",
    "#6db6ff",
    "#b6dbff",
    "#ff6db6",
    "#924900",
    "#db6d00",
    "#24ff24",
    "#ffff6d",
    "rgb(215,127,78)",
    "rgb(191,160,68)",
    "rgb(119,175,75)",
    "rgb(43,179,150)",
    "rgb(105,114,197)",
    "rgb(188,77,175)",
    "rgb(204,145,209)",
    "rgb(153,61,38)",
    "rgb(119,89,12)",
    "rgb(62,106,45)",
]


default_pallete = tab20_pallete


def set_default_colors(pallete):
    global default_pallete
    default_pallete = pallete


encoding_options = [
    "cl100k_base",
    "r50k_base",
    "p50k_base",
]

default_encoding = "cl100k_base"


class InteractiveTokenizer:
    def __init__(
        self,
        initial_text="",
        encoding=default_encoding,
        show_config=True,
        show_input=True,
        input_ui=None,
        show_stats=True,
        color_mapping=None,
    ):
        if not color_mapping:
            self.color_mapping = StableMapping(default_pallete)

        self.encoding = encoding
        self.view_mode = "simple"
        self.text = initial_text
        self.history = [self.text]
        self.input_ui, self.config_ui, self.stats_ui, self.output_ui, self.ui = (
            self._build_ui(
                input_ui=input_ui,
                show_config=show_config,
                show_input=show_input,
                show_stats=show_stats,
            )
        )
        self._trigger_ui_update()

    def _ipython_display_(self):
        display(css)
        display(self.ui)

    def _update_encoding(self, change):
        self.encoding = change["new"]
        self._reset_color_mapping()
        self._trigger_ui_update()

    def _update_view_mode(self, change):
        self.view_mode = change["new"]
        self._trigger_ui_update()

    def _reset_color_mapping(self):
        self.color_mapping.reset()

    def _update_text(self, change):
        self.text = change["new"]
        if self.text == "":
            self.clean_history()
        else:
            self.history.append(self.text)
            if len(self.history) > 10:
                self.history = self.history[-10:]
        self._trigger_ui_update()

    def _get_full_history_tokenids(self):
        tokenids = []
        for line in self.history:
            _, line_tokenids = split_tokens(line, self.encoding)
            tokenids.extend(line_tokenids)
        return tokenids

    def _trigger_ui_update(self):
        if self.view_mode == "simple":
            fragments, tokenids, rows, cols, colors = self._build_simple_matrix()
        elif self.view_mode == "history":
            fragments, tokenids, rows, cols, colors = self._build_history_matrix()
        elif self.view_mode == "multiline":
            fragments, tokenids, rows, cols, colors = self._build_multiline_matrix()

        self.output_ui.decoded = fragments
        self.output_ui.encoded = tokenids
        self.output_ui.rows = rows
        self.output_ui.cols = cols
        # For better color stability we use the history of tokenids
        self.color_mapping.map(self._get_full_history_tokenids())
        self.output_ui.colors = [self.color_mapping[t] for t in tokenids]
        if self.stats_ui:
            self.stats_ui.value = self._get_stats_html(fragments, tokenids)

    def _get_stats_html(self, fragments, _):
        n_tokens = len(fragments)
        n_words = len(self.text.split())
        return f"""<div><span class=\"tokenstats\">{n_tokens}</span> tokens</div>
        <div><span class=\"tokenstats\">{n_words}<span> words</div>
        <div><span class=\"tokenstats\">{round(n_tokens/n_words,1) if n_words else '-'}<span> t/w</div>"""

    def _build_simple_matrix(self):
        fragments, tokenids = split_tokens(self.text, self.encoding)
        return (
            fragments,
            tokenids,
            [1] * len(fragments),
            [*range(1, len(fragments) + 1)],
            None,
        )

    def _build_history_matrix(self):
        fragments = []
        tokenids = []
        rows = []
        cols = []
        for r, line in enumerate(reversed(self.history)):
            line_fragments, line_tokenids = split_tokens(line, self.encoding)
            fragments.extend(line_fragments)
            tokenids.extend(line_tokenids)
            rows.extend([r + 1] * len(line_fragments))
            cols.extend([*range(1, len(line_fragments) + 1)])

        return (fragments, tokenids, rows, cols, None)

    def _build_multiline_matrix(self):
        fragments = []
        tokenids = []
        rows = []
        cols = []
        for r, line in enumerate(self.text.splitlines(True)):
            line_fragments, line_tokenids = split_tokens(line, self.encoding)
            fragments.extend(line_fragments)
            tokenids.extend(line_tokenids)
            rows.extend([r + 1] * len(line_fragments))
            cols.extend([*range(1, len(line_fragments) + 1)])

        return (fragments, tokenids, rows, cols, None)

    def clean_history(self):
        self.history = []

    def _build_ui(
        self,
        input_ui=None,
        config_ui=None,
        show_config=True,
        show_input=True,
        show_stats=True,
    ):
        if input_ui is None and show_input:
            input_ui = widgets.Textarea(
                value=self.text,
                placeholder="Ingresa el texto aquí",
                description="Texto:",
                disabled=False,
                layout=Layout(width="100%", height="3rem"),
            )

        if input_ui:
            # If input_ui provided from another widget
            input_ui.observe(self._update_text, "value")

        if not show_input:
            # but show_input is False. We hide the input_ui but we make the observer
            input_ui = None

        output_ui = TokenMatrix()

        if show_stats:
            stats_ui = widgets.HTML(
                value="", layout=Layout(width="7rem", height="8rem")
            )
        else:
            stats_ui = None

        if config_ui is None and show_config:
            view_mode = widgets.RadioButtons(
                options=[
                    ("Simple", "simple"),
                    ("Historia", "history"),
                    ("Multilínea", "multiline"),
                ],
                value="simple",
            )

            encoding_dropdown = widgets.Dropdown(
                options=encoding_options,
                description="Tokenizer:",
                value=self.encoding,
            )

            encoding_dropdown.observe(self._update_encoding, "value")
            view_mode.observe(self._update_view_mode, "value")

            config_ui = Accordion(
                children=[VBox([view_mode, encoding_dropdown])],
                titles=("Configuración",),
            )
        elif not show_config:
            config_ui = None

        topbar = [x for x in (input_ui, stats_ui, config_ui) if x]
        bottombar = Box([output_ui], layout={"overflow": "scroll"})
        if len(topbar) >= 3:
            full_ui = VBox([HBox(topbar), bottombar])
        elif topbar:
            full_ui = HBox([output_ui] + topbar)
        else:
            full_ui = bottombar

        return (input_ui, config_ui, stats_ui, output_ui, full_ui)

    def get_synced_tokenizer(self, **kwargs):
        return InteractiveTokenizer(input_ui=self.input_ui, show_input=False, **kwargs)


show_tokens = partial(
    InteractiveTokenizer, show_input=False, show_config=False, show_stats=False
)


separator = widgets.HTML("<hr />")


def compare_encodings():
    x = InteractiveTokenizer()
    display(x)
    display(separator)
    display(x.get_synced_tokenizer(encoding="p50k_base"))
