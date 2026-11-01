"""Plot style shared by every figure of the paper, and the fixed name, colour and marker of each
system, identical across collections so a reader can follow a system from figure to figure.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

STYLE = {
    'palette': ["#EC8F8D", "#44A194", "#537D96"],
    'markers': ['o', 's', '^', 'D', 'P', 'X', 'v', '<'],
    'hatches': ['', '//', 'xx', '..', 'oo', '\\', '--', '++'],
    'figure_size': (7.5, 4.5),
    'figure_size_wide': (11.5, 4.5),
    'dpi': 150,
    'font_family': 'serif',
    'title_size': 18,
    'label_size': 18,
    'label_linespacing': 2.0,
    'tick_size': 16,
    'legend_size': 18,
    'line_width': 2.0,
    'marker_size': 8,
    'grid_alpha': 0.38,
    'save_format': 'pdf',
    'save_dpi': 300,
    'output_dir': Path('figures'),
    'bg_color': '#F4F0E4',
    'edge_color': '#333333',
    'colors': {
        'U':   '#EC8F8D',
        'SVB': '#44A194',
        'DVB': '#537D96',
    },
}

KUBRICK_CMAP = LinearSegmentedColormap.from_list(
    "kubrick", [STYLE['colors']['U'], STYLE['colors']['SVB'], STYLE['colors']['DVB']]
)


def color_for_kp(i, n):
    """Deterministic colour for series i out of n (common_plot_style.ipynb)."""
    if n == 1:
        return STYLE['colors']['U']
    if n <= 3:
        return [STYLE['colors']['U'], STYLE['colors']['SVB'], STYLE['colors']['DVB']][i]
    return KUBRICK_CMAP(i / (n - 1))


def configure():
    sns.set_theme(style='whitegrid', context='notebook', palette=STYLE['palette'])
    mpl.rcParams.update({
        'figure.figsize': STYLE['figure_size'],
        'figure.dpi': STYLE['dpi'],
        'savefig.dpi': STYLE['save_dpi'],
        'savefig.format': STYLE['save_format'],
        'savefig.bbox': 'tight',
        'savefig.transparent': False,
        'font.family': STYLE['font_family'],
        'font.size': STYLE['label_size'],
        'text.usetex': True,
        'axes.titlesize': STYLE['title_size'],
        'axes.labelsize': STYLE['label_size'],
        'axes.linewidth': 0.8,
        'axes.edgecolor': STYLE['edge_color'],
        'xtick.labelsize': STYLE['tick_size'],
        'xtick.major.width': 0.8,
        'ytick.labelsize': STYLE['tick_size'],
        'ytick.major.width': 0.8,
        'legend.fontsize': STYLE['legend_size'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.facecolor': STYLE['bg_color'],
        'axes.axisbelow': True,
        'grid.alpha': STYLE['grid_alpha'],
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
    })


def apply_axes_style(ax, title=None, xlabel=None, ylabel=None, xscale='linear', yscale='linear'):
    if title:
        ax.set_title(title, pad=10, weight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, **({'linespacing': STYLE['label_linespacing']}
                                 if '\n' in xlabel else {}))
    if ylabel:
        ax.set_ylabel(ylabel, **({'linespacing': STYLE['label_linespacing']}
                                 if '\n' in ylabel else {}))
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=STYLE['tick_size'])
    return ax


def save_figure(fig, name, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    full_path = output_path / f'{name}.{STYLE["save_format"]}'
    fig.savefig(full_path)
    return full_path


SERIES_ORDER = [
    "kannolo",
    "kannolo-rerank",
    "hnswlib",
    "glass",
    "symphonyqg",
    "lorann",
    "rabitq-hnsw",
    "vamana-leanvec(svs)",
]

DISPLAY = {
    "kannolo": r"kANNolo-HNSW",
    "kannolo-rerank": r"kANNolo-Rerank",
    "hnswlib": r"hnswlib",
    "glass": r"Glass",
    "symphonyqg": r"SymphonyQG",
    "lorann": r"LoRANN",
    "rabitq-hnsw": r"HNSW-RaBitQ",
    "vamana-leanvec(svs)": r"SVS-LeanVec",
}

SERIES_COLOUR = {
    "kannolo":             "#EC8F8D",  # house anchor 1
    "kannolo-rerank":      "#537D96",  # house anchor 3
    "hnswlib":             "#44A194",  # house anchor 2
    "glass":               "#E08A3C",  # orange
    "symphonyqg":          "#8E6FB0",  # purple
    "lorann":              "#C4A63C",  # ochre
    "rabitq-hnsw":         "#9C4F6B",  # wine
    "vamana-leanvec(svs)": "#4E6E9E",  # indigo
}

MARKER_SLOT = {algo: i for i, algo in enumerate(SERIES_ORDER)}

CONTROL_PARENT = {
    "kannolo-rerank-ext8": "kannolo-rerank",
    "kannolo-rerank-ext8-resid": "kannolo-rerank",
    "lorann-plain": "lorann",
}

CONTROL_HATCH = {
    "kannolo-rerank-ext8": "//",
    "kannolo-rerank-ext8-resid": "xx",
    "lorann-plain": "..",
}

CONTROL_MARKER = {
    "kannolo-rerank-ext8": "p",        # pentagon
    "kannolo-rerank-ext8-resid": "*",  # star -- must not be confusable with the pentagon above,
    "lorann-plain": "h",               # hexagon
}

DISPLAY.update({
    "kannolo-rerank-ext8": r"kANNolo-Rerank (ext-8)",
    "kannolo-rerank-ext8-resid": r"kANNolo-Rerank (ext-8, residual)",
    "lorann-plain": r"LoRANN",
})


SPARSE_SERIES_ORDER = [
    "kannolo-hnsw",
    "kannolo-hnsw-rerank",
    "seismic",
    "seismic-wave",
    "pyanns",
    "sindi",
    "lsp",
]

DISPLAY.update({
    "kannolo-hnsw": r"kANNolo-HNSW",
    "kannolo-hnsw-rerank": r"kANNolo-Rerank",
    "seismic": r"Seismic",
    "seismic-wave": r"Seismic-Wave",
    "pyanns": r"PyANN",
    "sindi": r"SINDI",
    "lsp": r"LSP",
})

SPARSE_SERIES_COLOUR = {
    "kannolo-hnsw":        "#EC8F8D",  # house anchor 1, as dense kannolo
    "kannolo-hnsw-rerank": "#537D96",  # house anchor 3, as dense kannolo-rerank
    "seismic":             "#44A194",  # house anchor 2
    "seismic-wave":        "#E08A3C",  # orange
    "pyanns":              "#8E6FB0",  # purple
    "sindi":               "#C4A63C",  # ochre
    "lsp":                 "#9C4F6B",  # wine
}

SPARSE_MARKER_SLOT = {algo: i for i, algo in enumerate(SPARSE_SERIES_ORDER)}


SPARSE_MODE = False


def use_sparse():
    """Point the series lookups at the sparse campaign's set.

    `series_colour`, `series_marker` and `SERIES_ORDER` are read by `p1_figures.py` through this
    module, so one rebinding here switches every figure at once rather than threading a `series`
    argument through each of them. Called once, from `p1_figures.main()`, before any figure is
    drawn -- never mid-run, or two figures in one invocation would disagree about what colour a
    library is.

    **GRAPH_BAND becomes every sparse system, and that is not a way of hiding anyone.** On dense
    it excludes two structural outliers (SymphonyQG at ~32x code replication, LoRANN which stores
    no copy of the collection) whose footprints stretch Fig. A's axis by a decade. The sparse
    field has no such outlier: all seven are one inverted index or one graph plus some
    representation of the collection, so the band is the whole field and the restriction is inert.
    """
    global SERIES_ORDER, SERIES_COLOUR, MARKER_SLOT, GRAPH_BAND, SPARSE_MODE
    SPARSE_MODE = True
    SERIES_ORDER = SPARSE_SERIES_ORDER
    SERIES_COLOUR = SPARSE_SERIES_COLOUR
    MARKER_SLOT = SPARSE_MARKER_SLOT
    GRAPH_BAND = list(SPARSE_SERIES_ORDER)

LEARN_SET_DATASETS = {"laion-clip-512-normalized"}

DISPLAY_ON_LEARN_SET = {
    "lorann": r"LoRANN (learned)",
}


def display(algo, dataset=None):
    """Series label, qualified where a library trained on data the others did not see."""
    if dataset in LEARN_SET_DATASETS and algo in DISPLAY_ON_LEARN_SET:
        return DISPLAY_ON_LEARN_SET[algo]
    return DISPLAY[algo]

GRAPH_BAND = [
    "kannolo",
    "kannolo-rerank",
    "hnswlib",
    "glass",
    "rabitq-hnsw",
    "vamana-leanvec(svs)",
]

REGIMES = {
    "index + one copy": GRAPH_BAND + ["lorann"],
    "codes per in-edge": ["symphonyqg"],
}

REGIME_ARMS = ["kannolo", "kannolo-rerank", "symphonyqg", "lorann"]

COLLECTION_SHAPE = {
    "agnews-mxbai-1024-euclidean":        (769_382, 1024),
    "msmarco-qwen-1024-normalized":       (8_840_823, 1024),
    "dpr-jina-768-normalized":            (20_969_760, 768),
    "laion-clip-512-normalized":          (1_000_448, 512),
    "hotpotqa-harrier-640-normalized":    (5_233_329, 640),
    "gooaq-distilroberta-768-normalized": (1_475_024, 768),
}


def collection_gib(dataset):
    """Raw f32 size of `dataset` in GiB, or None if we have no shape for it."""
    shape = COLLECTION_SHAPE.get(dataset)
    if shape is None:
        return None
    n, d = shape
    return n * d * 4 / 1024 ** 3


def series_colour(algo):
    return SERIES_COLOUR[CONTROL_PARENT.get(algo, algo)]


def series_marker(algo):
    if algo in CONTROL_MARKER:
        return CONTROL_MARKER[algo]
    return STYLE['markers'][MARKER_SLOT[algo] % len(STYLE['markers'])]


def series_hatch(algo):
    """Bar hatch. A control arm needs its OWN hatch, not the parent's: on D1 and D2
    `kannolo-rerank` has two controls, and inheriting the parent's slot drew both of them as the
    same hollow bar in the same colour with the same fill -- the same defect CONTROL_MARKER
    fixes for the scatter figures."""
    if algo in CONTROL_HATCH:
        return CONTROL_HATCH[algo]
    parent = CONTROL_PARENT.get(algo, algo)
    return STYLE['hatches'][MARKER_SLOT[parent] % len(STYLE['hatches'])]


def is_control_arm(algo):
    """Draw hollow: face unfilled, the parent's colour as the edge."""
    return algo in CONTROL_PARENT


def legend_under_axes(fig, handles, labels, ncol, gap_in=0.06, **kw):
    """One framed legend directly under the panels' x-labels, `gap_in` inches below them.

    Lay the panels out on the whole figure first (tight_layout without a reserved strip), then
    anchor the legend's top edge to the lowest point of any axes' decorations. Saved with
    bbox_inches='tight', the figure is cropped to the legend, so no blank band is left between
    the axes and the legend however many rows the legend takes.
    """
    fig.tight_layout()
    renderer = fig.canvas.get_renderer()
    y0 = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes) / fig.dpi  # inches
    y = (y0 - gap_in) / fig.get_figheight()
    return fig.legend(handles, labels, ncol=ncol, loc='upper center', bbox_to_anchor=(0.5, y),
                      frameon=True, fancybox=False, edgecolor='0.8',
                      fontsize=STYLE['legend_size'], **kw)


DISPLAY_TEX = {
    "kANNolo": r"\textsc{kANNolo}",
    "kANNolo-HNSW": r"\textsc{kANNolo}-\textsc{Hnsw}",
    "kANNolo-Rerank": r"\textsc{kANNolo}-\textsc{Rerank}",
    "kANNolo-Rerank (ext-8)": r"\textsc{kANNolo}-\textsc{Rerank} (ext-8)",
    "hnswlib": r"\texttt{hnswlib}",
    "Glass": r"\textsc{Glass}",
    "SymphonyQG": r"\textsc{SymphonyQG}",
    "LoRANN": r"\textsc{LoRANN}",
    "HNSW-RaBitQ": r"\textsc{Hnsw}-\textsc{RaBitQ}",
    "SVS-LeanVec": r"\textsc{Svs}-\textsc{LeanVec}",
    "Seismic": r"\textsc{Seismic}",
    "Seismic-Wave": r"\textsc{SeismicWave}",
    "PyANN": r"\textsc{PyANN}",
    "SINDI": r"\textsc{Sindi}",
    "LSP": r"\textsc{Lsp}",
    "Faiss": r"\textsc{Faiss}",
    "HNSW": r"\textsc{Hnsw}",
}


def tex_label(display):
    """`display` with the macro rendering of its system, if the paper has a macro for it."""
    return DISPLAY_TEX.get(display, display)
