#!/usr/bin/env python3
"""
Multiplex Channel Inventory — Interactive TUI
pip install textual pandas pyarrow
python cli.py
"""
from __future__ import annotations

import re

import pandas as pd
from rich.table import Table
from rich.table import box as rich_box
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    SelectionList,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from spatialprot_data._config import get_datasets_dir


# ── Constants ─────────────────────────────────────────────────────────────────

MULTIPLEX_MODALITIES = ("codex", "imc", "cycif")
UNIPROT_EXCLUDED_VALUES = {"", "exclude", "excluded", "none", "nan"}
DEFAULT_HEATMAP_PAIR_COUNT = 6
DEFAULT_HEATMAP_MARKER_LIMIT = 400
NUCLEAR_RULES: list[tuple[str, str]] = [
    (r"\bDAPI\b", "DAPI stain"),
    (r"HOECHST", "Hoechst stain"),
    (r"\bDNA[ _-]?\d*\b", "DNA stain"),
    (r"DSDNA", "dsDNA marker"),
    (r"191\s*IR|193\s*IR|IR191|IR193|IRIDIUM", "Iridium DNA intercalator"),
    (r"HISTONE\s*-?\s*H3|HISTONEH3|\bH3\b", "Histone H3 marker"),
    (r"NUCLEAR", "Explicit nuclear label"),
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def detect_nuclear_channels(names: pd.Series) -> tuple[pd.Series, pd.Series]:
    upper = names.fillna("").astype(str).str.upper()
    is_nuc = pd.Series(False, index=names.index)
    reason = pd.Series("", index=names.index, dtype="object")
    for pattern, label in NUCLEAR_RULES:
        m = upper.str.contains(pattern, regex=True, na=False) & ~is_nuc
        is_nuc.loc[m] = True
        reason.loc[m] = label
    return is_nuc, reason


def is_real_uniprot(v: object) -> bool:
    return not pd.isna(v) and str(v).strip().lower() not in UNIPROT_EXCLUDED_VALUES


def fmt_uniprot(v: object) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in UNIPROT_EXCLUDED_VALUES else s


def fmt_qc(v: object) -> str:
    if pd.isna(v):
        return "unknown"
    return "pass" if bool(v) else "fail"


def load_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets_dir = get_datasets_dir()
    rows: list[dict] = []
    mod_rows: list[dict] = []

    for dataset_dir in sorted(p for p in datasets_dir.iterdir() if p.is_dir()):
        dname = dataset_dir.name
        for modality in MULTIPLEX_MODALITIES:
            path = dataset_dir / modality / "channels.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path).copy()
            if "channel_name" not in df.columns:
                continue
            df["channel_name"] = df["channel_name"].astype(str)
            for col in ("uniprot_id", "qc_pass", "description"):
                if col not in df.columns:
                    df[col] = pd.NA

            nuc, reason = detect_nuclear_channels(df["channel_name"])
            has_uni = df["uniprot_id"].map(is_real_uniprot)

            ddf = pd.DataFrame({
                "dataset": dname, "modality": modality,
                "channel_index": range(len(df)),
                "channel_name": df["channel_name"],
                "uniprot_id": df["uniprot_id"],
                "qc_pass": df["qc_pass"],
                "description": df["description"],
                "is_nuclear": nuc,
                "nuclear_reason": reason,
                "has_uniprot": has_uni,
                "channels_path": str(path),
            })
            rows.extend(ddf.to_dict(orient="records"))
            mod_rows.append({
                "dataset": dname, "modality": modality,
                "num_channels": len(ddf),
                "num_nuclear": int(ddf["is_nuclear"].sum()),
                "num_uniprot": int(ddf["has_uniprot"].sum()),
                "num_qc_pass": int(pd.Series(ddf["qc_pass"]).fillna(False).astype(bool).sum()),
            })

    ch = pd.DataFrame(rows)
    sm = pd.DataFrame(mod_rows)
    if not ch.empty:
        ch["qc_display"] = ch["qc_pass"].map(fmt_qc)
        ch["uni_display"] = ch["uniprot_id"].map(fmt_uniprot)
        ch["dm"] = ch["dataset"] + " / " + ch["modality"]
    return ch, sm


def apply_common_filters(
    df: pd.DataFrame,
    qc: str,
    nuclear_only: bool,
    mapped_only: bool,
    search: str,
) -> pd.DataFrame:
    out = df.copy()
    if qc != "all":
        out = out[out["qc_display"] == qc]
    if nuclear_only:
        out = out[out["is_nuclear"]]
    if mapped_only:
        out = out[out["has_uniprot"]]
    if search:
        pat = re.escape(search)
        mask = (
            out["channel_name"].str.contains(pat, case=False, na=False)
            | out["uni_display"].str.contains(pat, case=False, na=False)
            | out["description"].fillna("").astype(str).str.contains(pat, case=False, na=False)
        )
        out = out[mask]
    return out


def apply_filters(
    df: pd.DataFrame,
    datasets: list[str],
    modalities: list[str],
    qc: str,
    nuclear_only: bool,
    mapped_only: bool,
    search: str,
) -> pd.DataFrame:
    out = df[df["dataset"].isin(datasets) & df["modality"].isin(modalities)].copy()
    return apply_common_filters(
        out,
        qc=qc,
        nuclear_only=nuclear_only,
        mapped_only=mapped_only,
        search=search,
    )


# ── Messages ──────────────────────────────────────────────────────────────────

class FiltersChanged(Message):
    pass


class ToggleSelectionList(SelectionList[str]):
    BINDINGS = [*SelectionList.BINDINGS, Binding("enter", "select", "Toggle option", show=False)]


# ── FilterPanel ───────────────────────────────────────────────────────────────

class FilterPanel(Widget):
    DEFAULT_CSS = """
    FilterPanel {
        width: 32;
        height: 100%;
        border-right: solid $primary-darken-2;
        background: $surface-darken-1;
        overflow-y: auto;
        padding: 0 1;
    }
    FilterPanel .section-label {
        color: $warning;
        text-style: bold;
        margin-top: 1;
    }
    FilterPanel ToggleSelectionList {
        height: auto;
        max-height: 10;
        border: solid $primary-darken-3;
        margin-bottom: 1;
    }
    FilterPanel Select {
        margin-bottom: 1;
    }
    FilterPanel Input {
        margin-bottom: 1;
    }
    FilterPanel .toggle-row {
        height: 3;
        align: left middle;
    }
    FilterPanel .toggle-label {
        margin-left: 1;
        color: $text;
    }
    """

    def __init__(
        self,
        datasets: list[str],
        dataset_modalities: dict[str, list[str]],
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._datasets = datasets
        self._dataset_modalities = dataset_modalities
        self._suspend_filter_events = False

    def compose(self) -> ComposeResult:
        yield Label("▸ DATASETS", classes="section-label")
        yield ToggleSelectionList(
            *[(dataset, dataset, index == 0) for index, dataset in enumerate(self._datasets)],
            id="f-datasets",
        )
        yield Label("[dim]Dataset and modality filters are single-select. Click, space, or enter to switch.[/]")
        yield Label("▸ MODALITIES", classes="section-label")
        yield ToggleSelectionList(id="f-mods")
        yield Label("▸ QC STATUS", classes="section-label")
        yield Select(
            [("All", "all"), ("Pass", "pass"), ("Fail", "fail"), ("Unknown", "unknown")],
            value="all",
            id="f-qc",
        )
        yield Label("▸ SEARCH", classes="section-label")
        yield Input(placeholder="name / uniprot / description…", id="f-search")
        with Horizontal(classes="toggle-row"):
            yield Switch(value=False, id="f-nuclear")
            yield Label("Nuclear only", classes="toggle-label")
        with Horizontal(classes="toggle-row"):
            yield Switch(value=False, id="f-mapped")
            yield Label("UniProt only", classes="toggle-label")

    def on_mount(self) -> None:
        self._sync_modality_options()

    def _current_dataset(self) -> str | None:
        selected = list(self.query_one("#f-datasets", ToggleSelectionList).selected)
        return selected[0] if selected else None

    def _ordered_modalities(self, dataset: str | None) -> list[str]:
        if dataset is None:
            return []
        available = self._dataset_modalities.get(dataset, [])
        return [modality for modality in MULTIPLEX_MODALITIES if modality in available]

    def _set_single_selection(self, widget_id: str, value: str) -> None:
        selection_list = self.query_one(widget_id, ToggleSelectionList)
        self._suspend_filter_events = True
        try:
            selection_list.deselect_all()
            selection_list.select(value)
        finally:
            self._suspend_filter_events = False

    def _sync_modality_options(self) -> None:
        dataset = self._current_dataset()
        available_modalities = self._ordered_modalities(dataset)
        modality_list = self.query_one("#f-mods", ToggleSelectionList)

        try:
            current_selected = list(modality_list.selected)
        except Exception:
            current_selected = []

        selected_modality = current_selected[0] if current_selected and current_selected[0] in available_modalities else None
        if selected_modality is None and available_modalities:
            selected_modality = available_modalities[0]

        self._suspend_filter_events = True
        try:
            modality_list.clear_options()
            for modality in available_modalities:
                modality_list.add_option((modality, modality, modality == selected_modality))
        finally:
            self._suspend_filter_events = False

    @on(ToggleSelectionList.SelectionToggled, "#f-datasets")
    def _dataset_toggled(self, event: ToggleSelectionList.SelectionToggled) -> None:
        if self._suspend_filter_events:
            return
        self._set_single_selection("#f-datasets", str(event.selection.value))
        self._sync_modality_options()
        self.post_message(FiltersChanged())

    @on(ToggleSelectionList.SelectionToggled, "#f-mods")
    def _modality_toggled(self, event: ToggleSelectionList.SelectionToggled) -> None:
        if self._suspend_filter_events:
            return
        self._set_single_selection("#f-mods", str(event.selection.value))
        self.post_message(FiltersChanged())

    @on(Select.Changed)
    @on(Switch.Changed)
    @on(Input.Changed)
    def _changed(self, _: object = None) -> None:
        if self._suspend_filter_events:
            return
        self.post_message(FiltersChanged())

    @property
    def selected_datasets(self) -> list[str]:
        selected = list(self.query_one("#f-datasets", ToggleSelectionList).selected)
        return selected[:1]

    @property
    def selected_modalities(self) -> list[str]:
        selected = list(self.query_one("#f-mods", ToggleSelectionList).selected)
        return selected[:1]

    @property
    def qc_filter(self) -> str:
        v = self.query_one("#f-qc", Select).value
        return str(v) if v else "all"

    @property
    def nuclear_only(self) -> bool:
        return bool(self.query_one("#f-nuclear", Switch).value)

    @property
    def mapped_only(self) -> bool:
        return bool(self.query_one("#f-mapped", Switch).value)

    @property
    def search(self) -> str:
        return self.query_one("#f-search", Input).value.strip()


# ── SummaryView ───────────────────────────────────────────────────────────────

class SummaryView(Widget):
    DEFAULT_CSS = """
    SummaryView {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        padding: 1;
    }
    SummaryView .metrics-row {
        height: 5;
        margin-bottom: 1;
    }
    SummaryView .metric-box {
        border: solid $primary-darken-1;
        width: 1fr;
        height: 5;
        content-align: center middle;
        margin: 0 1;
        background: $surface;
    }
    SummaryView DataTable {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(classes="metrics-row"):
            yield Static("", id="m-pairs", classes="metric-box")
            yield Static("", id="m-datasets", classes="metric-box")
            yield Static("", id="m-channels", classes="metric-box")
            yield Static("", id="m-nuclear", classes="metric-box")
            yield Static("", id="m-uniprot", classes="metric-box")
        yield DataTable(id="summary-table", cursor_type="row")

    def refresh_data(self, filtered: pd.DataFrame, summary: pd.DataFrame) -> None:
        def metric(val: int, label: str) -> str:
            return f"[bold white]{val}[/]\n[dim]{label}[/]"

        pairs = int(filtered[["dataset", "modality"]].drop_duplicates().shape[0])
        self.query_one("#m-pairs", Static).update(metric(pairs, "pairs"))
        self.query_one("#m-datasets", Static).update(metric(filtered["dataset"].nunique(), "datasets"))
        self.query_one("#m-channels", Static).update(metric(len(filtered), "channels"))
        self.query_one("#m-nuclear", Static).update(metric(int(filtered["is_nuclear"].sum()), "nuclear"))
        self.query_one("#m-uniprot", Static).update(metric(int(filtered["has_uniprot"].sum()), "uniprot"))

        tbl = self.query_one("#summary-table", DataTable)
        tbl.clear(columns=True)

        if summary.empty or filtered.empty:
            return

        visible = filtered[["dataset", "modality"]].drop_duplicates()
        vis = (
            summary.merge(visible, on=["dataset", "modality"], how="inner")
            .sort_values(["dataset", "modality"])
            .reset_index(drop=True)
        )
        tbl.add_columns(*[c.replace("_", " ").title() for c in vis.columns])
        for _, row in vis.iterrows():
            tbl.add_row(*[str(v) for v in row])


# ── ExplorerView ──────────────────────────────────────────────────────────────

class ExplorerView(Widget):
    DEFAULT_CSS = """
    ExplorerView {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        padding: 1;
    }
    ExplorerView Select {
        margin-bottom: 1;
    }
    ExplorerView #explorer-body {
        height: 1fr;
    }
    ExplorerView #explorer-table {
        width: 3fr;
        height: 100%;
    }
    ExplorerView #explorer-side {
        width: 1fr;
        height: 100%;
        padding: 0 1;
        border-left: solid $primary-darken-2;
    }
    ExplorerView #explorer-side .stat-line {
        color: $text;
        margin-bottom: 0;
    }
    ExplorerView #explorer-nuclear {
        height: 1fr;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Select([], id="explorer-pair", prompt="Select dataset / modality…")
        with Horizontal(id="explorer-body"):
            yield DataTable(id="explorer-table", cursor_type="row")
            with Vertical(id="explorer-side"):
                yield Label("[bold]Pair summary[/]")
                yield Static("", id="explorer-stats")
                yield Label("[bold]Nuclear channels[/]")
                yield DataTable(id="explorer-nuclear", cursor_type="row")

    def refresh_pairs(self, pairs: list[str], current: str | None) -> None:
        sel = self.query_one("#explorer-pair", Select)
        opts = [(p, p) for p in pairs]
        sel.set_options(opts)
        if current and current in pairs:
            sel.value = current
        elif pairs:
            sel.value = pairs[0]

    @on(Select.Changed, "#explorer-pair")
    def _pair_changed(self, event: Select.Changed) -> None:
        if event.value and hasattr(self, "_filtered_df"):
            self._render_pair(str(event.value))

    def refresh_data(self, filtered: pd.DataFrame) -> None:
        self._filtered_df = filtered
        pairs = sorted(filtered["dm"].unique())
        current = None
        try:
            v = self.query_one("#explorer-pair", Select).value
            current = str(v) if v else None
        except Exception:
            pass
        self.refresh_pairs(pairs, current)
        if pairs:
            pair = current if current and current in pairs else pairs[0]
            self._render_pair(pair)

    def _render_pair(self, pair: str) -> None:
        df = self._filtered_df
        pair_df = df[df["dm"] == pair].sort_values("channel_index").reset_index(drop=True)

        # Main table
        tbl = self.query_one("#explorer-table", DataTable)
        tbl.clear(columns=True)
        cols = ["channel_index", "channel_name", "uni_display", "qc_display", "is_nuclear", "nuclear_reason", "description"]
        labels = ["#", "Channel", "UniProt", "QC", "Nuclear", "Reason", "Description"]
        tbl.add_columns(*labels)
        for _, row in pair_df.iterrows():
            style = "on dark_goldenrod" if row["is_nuclear"] else ""
            cells = [str(row.get(c, "")) for c in cols]
            tbl.add_row(*cells, key=str(row["channel_index"]))
        if pair_df["is_nuclear"].any():
            # Re-render with colour highlights via Text objects
            tbl.clear(columns=True)
            tbl.add_columns(*labels)
            for _, row in pair_df.iterrows():
                cells = []
                for c in cols:
                    val = str(row.get(c, ""))
                    if row["is_nuclear"]:
                        cells.append(Text(val, style="bold yellow"))
                    elif c == "uni_display" and not row["has_uniprot"]:
                        cells.append(Text(val, style="red"))
                    else:
                        cells.append(val)
                tbl.add_row(*cells)

        # Side stats
        self.query_one("#explorer-stats", Static).update(
            f"Channels: [bold]{len(pair_df)}[/]\n"
            f"Nuclear:  [bold]{int(pair_df['is_nuclear'].sum())}[/]\n"
            f"UniProt:  [bold]{int(pair_df['has_uniprot'].sum())}[/]\n"
            f"QC pass:  [bold]{int((pair_df['qc_display'] == 'pass').sum())}[/]\n\n"
            f"[dim]{pair_df['channels_path'].iloc[0] if len(pair_df) else ''}[/]"
        )

        # Nuclear sub-table
        ntbl = self.query_one("#explorer-nuclear", DataTable)
        ntbl.clear(columns=True)
        ndf = pair_df[pair_df["is_nuclear"]][["channel_name", "nuclear_reason", "uni_display"]]
        if ndf.empty:
            ntbl.add_columns("Status")
            ntbl.add_row("None detected")
        else:
            ntbl.add_columns("Channel", "Reason", "UniProt")
            for _, row in ndf.iterrows():
                ntbl.add_row(row["channel_name"], row["nuclear_reason"], row["uni_display"])


# ── NuclearView ───────────────────────────────────────────────────────────────

class NuclearView(Widget):
    DEFAULT_CSS = """
    NuclearView {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        padding: 1;
    }
    NuclearView DataTable {
        height: 1fr;
    }
    NuclearView Label {
        margin-bottom: 1;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Nuclear channels detected by heuristic rules across all filtered datasets.")
        yield DataTable(id="nuclear-table", cursor_type="row")

    def refresh_data(self, filtered: pd.DataFrame) -> None:
        tbl = self.query_one("#nuclear-table", DataTable)
        tbl.clear(columns=True)

        ndf = filtered[filtered["is_nuclear"]].copy()
        if ndf.empty:
            tbl.add_columns("Status")
            tbl.add_row("No nuclear channels match current filters.")
            return

        pivot = (
            ndf.groupby(["dataset", "modality"], as_index=False)
            .agg(
                channels=("channel_name", lambda s: ", ".join(s.astype(str))),
                count=("channel_name", "size"),
                reasons=("nuclear_reason", lambda s: ", ".join(s.unique())),
            )
            .sort_values(["dataset", "modality"])
        )
        tbl.add_columns("Dataset", "Modality", "Count", "Channels", "Detection reason")
        for _, row in pivot.iterrows():
            tbl.add_row(
                row["dataset"], row["modality"],
                str(row["count"]), row["channels"], row["reasons"],
            )


# ── HeatmapView ───────────────────────────────────────────────────────────────

class HeatmapView(Widget):
    DEFAULT_CSS = """
    HeatmapView {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        padding: 1;
    }
    HeatmapView #hmap-controls {
        width: 36;
        height: 100%;
        border-right: solid $primary-darken-2;
        padding: 0 1;
        overflow-y: auto;
    }
    HeatmapView #hmap-controls .ctrl-label {
        color: $warning;
        text-style: bold;
        margin-top: 1;
    }
    HeatmapView #hmap-controls SelectionList {
        height: auto;
        max-height: 16;
        border: solid $primary-darken-3;
        margin-bottom: 1;
    }
    HeatmapView #hmap-controls Select {
        margin-bottom: 1;
    }
    HeatmapView #hmap-controls .toggle-row {
        height: 3;
        align: left middle;
    }
    HeatmapView #hmap-controls .toggle-label {
        margin-left: 1;
    }
    HeatmapView #hmap-right {
        width: 1fr;
        height: 100%;
    }
    HeatmapView #hmap-grid {
        padding: 1;
    }
    HeatmapView #hmap-legend {
        height: 3;
        padding: 0 2;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="hmap-controls"):
                yield Label("▸ COMPARE DATASETS", classes="ctrl-label")
                yield Label("[dim]Click, space, or enter to toggle datasets.[/]")
                yield ToggleSelectionList(id="hmap-datasets")
                yield Label("▸ MATCH MARKERS BY", classes="ctrl-label")
                yield Select(
                    [
                        ("Channel name", "name"),
                        ("UniProt ID (fallback to name)", "uniprot"),
                        ("Both (name + UniProt)", "both"),
                    ],
                    value="both",
                    id="hmap-match",
                )
                yield Label("▸ SORT MARKERS BY", classes="ctrl-label")
                yield Select(
                    [("Alphabetical", "alpha"), ("Prevalence (most common first)", "prev")],
                    value="prev",
                    id="hmap-sort",
                )
                yield Label("▸ MAX MARKERS", classes="ctrl-label")
                yield Select(
                    [("100", "100"), ("250", "250"), ("400", "400"), ("800", "800"), ("All", "all")],
                    value=str(DEFAULT_HEATMAP_MARKER_LIMIT),
                    id="hmap-max-markers",
                )
                with Horizontal(classes="toggle-row"):
                    yield Switch(value=False, id="hmap-excl-nuclear")
                    yield Label("Exclude nuclear channels", classes="toggle-label")

            with Vertical(id="hmap-right"):
                yield Static("", id="hmap-legend")
                with ScrollableContainer():
                    yield Static("", id="hmap-grid")

    def refresh_data(self, filtered: pd.DataFrame) -> None:
        self._filtered_df = filtered
        pairs = sorted(filtered["dm"].unique())

        sl = self.query_one("#hmap-datasets", ToggleSelectionList)
        try:
            previously = set(sl.selected)
        except Exception:
            previously = set()

        default_selected = set(pairs[:DEFAULT_HEATMAP_PAIR_COUNT])
        selected_pairs = previously or default_selected

        self._suspend_heatmap_render = True
        try:
            sl.clear_options()
            for p in pairs:
                sl.add_option((p, p, p in selected_pairs))
        finally:
            self._suspend_heatmap_render = False

        if self._is_active_tab():
            self._render_heatmap()
        else:
            self.query_one("#hmap-grid", Static).update("[dim]Open the Heatmap tab to render the comparison.[/]")
            self.query_one("#hmap-legend", Static).update("")

    def _is_active_tab(self) -> bool:
        return self.app.query_one("#tabs", TabbedContent).active == "heatmap"

    @on(ToggleSelectionList.SelectedChanged, "#hmap-datasets")
    @on(Select.Changed, "#hmap-match")
    @on(Select.Changed, "#hmap-sort")
    @on(Select.Changed, "#hmap-max-markers")
    @on(Switch.Changed, "#hmap-excl-nuclear")
    def _controls_changed(self, _: object = None) -> None:
        if getattr(self, "_suspend_heatmap_render", False):
            return
        if hasattr(self, "_filtered_df") and self._is_active_tab():
            self._render_heatmap()

    def _render_heatmap(self) -> None:
        filtered = self._filtered_df

        selected_pairs = list(self.query_one("#hmap-datasets", ToggleSelectionList).selected)
        match_by = self.query_one("#hmap-match", Select).value or "both"
        sort_by = self.query_one("#hmap-sort", Select).value or "prev"
        max_markers = self.query_one("#hmap-max-markers", Select).value or str(DEFAULT_HEATMAP_MARKER_LIMIT)
        excl_nuclear = self.query_one("#hmap-excl-nuclear", Switch).value

        grid = self.query_one("#hmap-grid", Static)
        legend = self.query_one("#hmap-legend", Static)

        if not selected_pairs:
            grid.update("[dim]Select at least one dataset to compare.[/]")
            legend.update("")
            return

        sub = filtered[filtered["dm"].isin(selected_pairs)].copy()
        if excl_nuclear:
            sub = sub[~sub["is_nuclear"]]
        if sub.empty:
            grid.update("[dim]No channels remain after filters.[/]")
            legend.update("")
            return

        # Build marker label
        match_by_str = str(match_by)
        if match_by_str == "name":
            sub["_label"] = sub["channel_name"]
        elif match_by_str == "uniprot":
            sub["_label"] = sub["uni_display"].where(sub["uni_display"] != "", sub["channel_name"])
        else:
            sub["_label"] = sub["channel_name"]
            has_uni = sub["uni_display"] != ""
            sub.loc[has_uni, "_label"] = sub.loc[has_uni, "channel_name"] + " (" + sub.loc[has_uni, "uni_display"] + ")"

        deduped = sub.drop_duplicates(subset=["dm", "_label"])
        matrix = (
            deduped.assign(_p=1)
            .pivot_table(index="_label", columns="dm", values="_p", aggfunc="max", fill_value=0)
        )
        # Ensure all selected pairs appear as columns (even if no channels)
        for p in selected_pairs:
            if p not in matrix.columns:
                matrix[p] = 0
        matrix = matrix[selected_pairs]

        sort_by_str = str(sort_by)
        if sort_by_str == "prev":
            matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]
        else:
            matrix = matrix.sort_index()

        max_markers_str = str(max_markers)
        truncated = False
        if max_markers_str != "all":
            limit = int(max_markers_str)
            if len(matrix) > limit:
                matrix = matrix.iloc[:limit]
                truncated = True

        # Build Rich table
        t = Table(
            box=rich_box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold #f0c040",
            padding=(0, 1),
            collapse_padding=True,
        )
        t.add_column("Marker", style="white", no_wrap=True, min_width=24)

        short_names = []
        for col in matrix.columns:
            short = col if len(col) <= 18 else col[:15] + "…"
            short_names.append(short)
            t.add_column(short, justify="center", no_wrap=True, min_width=4)

        present_text = Text("█", style="bold #4499ff")
        absent_text  = Text("·", style="#444444")

        for marker in matrix.index:
            cells: list[str | Text] = [marker]
            for col in matrix.columns:
                cells.append(Text("█", style="bold #4499ff") if matrix.at[marker, col] else Text("·", style="#444444"))
            t.add_row(*cells)

        n_markers = len(matrix)
        n_present_total = int(matrix.values.sum())
        n_cells = n_markers * len(selected_pairs)

        grid.update(t)
        trunc_msg = f"  │  showing first {n_markers} markers" if truncated else ""
        legend.update(
            f"[bold #4499ff]█[/] present   [#444444]·[/] absent   "
            f"[dim]│  {n_markers} markers  │  {len(selected_pairs)} datasets  │  "
            f"{n_present_total}/{n_cells} cells filled ({100 * n_present_total // max(n_cells, 1)}%){trunc_msg}[/]"
        )


# ── Main App ──────────────────────────────────────────────────────────────────

class ChannelInventoryApp(App):
    TITLE = "Multiplex Channel Inventory"
    CSS = """
    Screen {
        layout: horizontal;
    }
    #main-content {
        width: 1fr;
        height: 100%;
    }
    TabbedContent {
        height: 100%;
    }
    TabPane {
        layout: vertical;
        width: 1fr;
        height: 1fr;
        padding: 0;
    }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+r", "reload", "Reload data"),
        Binding("1", "switch_tab('summary')", "Summary", show=False),
        Binding("2", "switch_tab('explorer')", "Explorer", show=False),
        Binding("3", "switch_tab('nuclear')", "Nuclear", show=False),
        Binding("4", "switch_tab('heatmap')", "Heatmap", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._channels_df = pd.DataFrame()
        self._summary_df = pd.DataFrame()
        self._filtered_df = pd.DataFrame()
        self._heatmap_df = pd.DataFrame()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Footer()

        # Load data synchronously before composing the main layout
        self._channels_df, self._summary_df = load_inventory()
        datasets = sorted(self._channels_df["dataset"].unique()) if not self._channels_df.empty else []
        dataset_modalities = {
            dataset: sorted(self._channels_df.loc[self._channels_df["dataset"] == dataset, "modality"].unique())
            for dataset in datasets
        }

        with Horizontal():
            yield FilterPanel(datasets=datasets, dataset_modalities=dataset_modalities, id="filter-panel")
            with Vertical(id="main-content"):
                with TabbedContent(id="tabs"):
                    with TabPane("📊 Summary", id="summary"):
                        yield SummaryView(id="summary-view")
                    with TabPane("🔍 Explorer", id="explorer"):
                        yield ExplorerView(id="explorer-view")
                    with TabPane("☢  Nuclear", id="nuclear"):
                        yield NuclearView(id="nuclear-view")
                    with TabPane("🔥 Heatmap", id="heatmap"):
                        yield HeatmapView(id="heatmap-view")

    def on_mount(self) -> None:
        if self._channels_df.empty:
            self.notify("No channel data found. Check your datasets directory.", severity="error")
            return
        self._refresh_all()

    @on(FiltersChanged)
    def _on_filter_changed(self, _: FiltersChanged) -> None:
        self._refresh_all()

    @on(TabbedContent.TabActivated, "#tabs")
    def _on_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.tab.id == "heatmap" and not self._heatmap_df.empty:
            self.query_one("#heatmap-view", HeatmapView).refresh_data(self._heatmap_df)

    def _refresh_all(self) -> None:
        if self._channels_df.empty:
            return
        fp = self.query_one("#filter-panel", FilterPanel)
        self._filtered_df = apply_filters(
            self._channels_df,
            datasets=fp.selected_datasets,
            modalities=fp.selected_modalities,
            qc=fp.qc_filter,
            nuclear_only=fp.nuclear_only,
            mapped_only=fp.mapped_only,
            search=fp.search,
        )
        self._heatmap_df = apply_common_filters(
            self._channels_df,
            qc=fp.qc_filter,
            nuclear_only=fp.nuclear_only,
            mapped_only=fp.mapped_only,
            search=fp.search,
        )
        self.query_one("#summary-view", SummaryView).refresh_data(self._filtered_df, self._summary_df)
        self.query_one("#explorer-view", ExplorerView).refresh_data(self._filtered_df)
        self.query_one("#nuclear-view", NuclearView).refresh_data(self._filtered_df)
        self.query_one("#heatmap-view", HeatmapView).refresh_data(self._heatmap_df)

    def action_reload(self) -> None:
        self.notify("Reloading data…")
        self._channels_df, self._summary_df = load_inventory()
        self._refresh_all()
        self.notify("Data reloaded.", severity="information")

    def action_switch_tab(self, tab: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab


if __name__ == "__main__":
    ChannelInventoryApp().run()