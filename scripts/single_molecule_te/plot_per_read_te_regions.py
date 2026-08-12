from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


CALL_COLUMNS = [0, 1, 2, 4, 6, 7, 8, 9]
CALL_NAMES = ["chrom", "pos", "strand", "read_id", "prob0", "prob1", "call", "kmer"]
CALL_DTYPES = {
    0: "string",
    1: "int64",
    2: "string",
    4: "string",
    6: "float32",
    7: "float32",
    8: "int8",
    9: "string",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-read methylation across reference-matched TE-rich regions."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("fig5_te_regions.tsv"),
        help="Tab-separated region configuration; paths are relative to --data-root.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bins", type=int, default=220)
    parser.add_argument("--max-reads", type=int, default=120)
    parser.add_argument(
        "--wgbs-min-coverage",
        type=int,
        default=5,
        help="Minimum WGBS coverage retained before binning.",
    )
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    return parser.parse_args()


def load_config(path: Path, data_root: Path) -> List[Dict[str, object]]:
    config = pd.read_csv(path, sep="\t", dtype="string")
    required = {
        "region_id",
        "species_label",
        "chrom",
        "start",
        "end",
        "call_file",
        "te_bed",
        "wgbs_beds",
    }
    missing = required.difference(config.columns)
    if missing:
        raise ValueError(f"Missing config columns: {', '.join(sorted(missing))}")

    rows: List[Dict[str, object]] = []
    for record in config.to_dict(orient="records"):
        start = int(record["start"])
        end = int(record["end"])
        if start >= end:
            raise ValueError(f"Invalid interval for {record['region_id']}: {start}-{end}")
        wgbs_paths = [
            data_root / value
            for value in str(record["wgbs_beds"]).split(";")
            if value.strip()
        ]
        rows.append(
            {
                "region_id": str(record["region_id"]),
                "species_label": str(record["species_label"]),
                "chrom": str(record["chrom"]),
                "start": start,
                "end": end,
                "call_file": data_root / str(record["call_file"]),
                "te_bed": data_root / str(record["te_bed"]),
                "wgbs_beds": wgbs_paths,
            }
        )
    return rows


def validate_inputs(regions: List[Dict[str, object]]) -> None:
    paths: List[Path] = []
    for region in regions:
        paths.extend([Path(region["call_file"]), Path(region["te_bed"])])
        paths.extend(Path(path) for path in region["wgbs_beds"])
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required input files were not found:\n{formatted}")


def extract_region_calls(
    region: Dict[str, object], cache_dir: Path, chunksize: int
) -> pd.DataFrame:
    region_id = str(region["region_id"])
    chrom = str(region["chrom"])
    start = int(region["start"])
    end = int(region["end"])
    cache_path = cache_dir / f"{region_id}_{chrom}_{start}_{end}.pkl"
    if cache_path.exists():
        return pd.read_pickle(cache_path)

    chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        region["call_file"],
        sep="\t",
        header=None,
        usecols=CALL_COLUMNS,
        names=CALL_NAMES,
        dtype=CALL_DTYPES,
        chunksize=chunksize,
        low_memory=False,
    ):
        subset = chunk[
            (chunk["chrom"] == chrom)
            & (chunk["pos"] >= start)
            & (chunk["pos"] < end)
        ].copy()
        if not subset.empty:
            chunks.append(subset)

    if not chunks:
        raise RuntimeError(f"No calls found in {region_id}: {chrom}:{start}-{end}")
    calls = pd.concat(chunks, ignore_index=True)
    calls.to_pickle(cache_path)
    return calls


def select_reads(calls: pd.DataFrame, max_reads: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    read_stats = (
        calls.groupby("read_id", as_index=False)
        .agg(n_calls=("prob1", "size"), mean_prob1=("prob1", "mean"))
        .sort_values(["n_calls", "read_id"], ascending=[False, True], kind="mergesort")
    )
    selected_ids = read_stats.head(max_reads)["read_id"]
    selected = calls[calls["read_id"].isin(selected_ids)].copy()
    return selected, read_stats.head(max_reads).copy()


def build_read_bin_matrix(
    selected: pd.DataFrame, start: int, end: int, n_bins: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = selected.copy()
    bin_width = (end - start) / n_bins
    selected["bin"] = np.floor((selected["pos"] - start) / bin_width).astype(int)
    selected["bin"] = selected["bin"].clip(0, n_bins - 1)

    matrix = selected.groupby(["read_id", "bin"])["prob1"].mean().unstack()
    matrix = matrix.reindex(columns=range(n_bins))
    order = (
        matrix.mean(axis=1, skipna=True)
        .rename("mean_prob1")
        .reset_index()
        .sort_values(["mean_prob1", "read_id"], ascending=[False, True], kind="mergesort")
    )
    matrix = matrix.reindex(order["read_id"])

    bin_table = pd.DataFrame(
        {
            "bin": np.arange(n_bins),
            "bin_start": start + np.arange(n_bins) * bin_width,
            "bin_end": start + (np.arange(n_bins) + 1) * bin_width,
        }
    )
    bin_table["bin_center"] = (bin_table["bin_start"] + bin_table["bin_end"]) / 2
    return matrix, bin_table


def load_wgbs_region(
    paths: List[Path], chrom: str, start: int, end: int, min_coverage: int
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        for chunk in pd.read_csv(
            path,
            sep="\t",
            header=None,
            usecols=[0, 1, 9, 10],
            names=["chrom", "pos", "coverage", "pct"],
            dtype={0: "string", 1: "int64", 9: "int64", 10: "float32"},
            chunksize=500_000,
            low_memory=False,
        ):
            subset = chunk[
                (chunk["chrom"] == chrom)
                & (chunk["pos"] >= start)
                & (chunk["pos"] < end)
                & (chunk["coverage"] >= min_coverage)
            ].copy()
            if not subset.empty:
                frames.append(subset)
    if not frames:
        return pd.DataFrame(columns=["chrom", "pos", "coverage", "beta"])
    wgbs = pd.concat(frames, ignore_index=True)
    wgbs["beta"] = wgbs["pct"] / 100.0
    return wgbs[["chrom", "pos", "coverage", "beta"]]


def load_te_region(
    path: Path, chrom: str, start: int, end: int
) -> tuple[pd.DataFrame, List[tuple[int, int]]]:
    te = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 7],
        names=["chrom", "start", "end", "class"],
        low_memory=False,
    )
    te = te[
        (te["chrom"] == chrom) & (te["end"] > start) & (te["start"] < end)
    ].copy()
    te["plot_start"] = te["start"].clip(lower=start)
    te["plot_end"] = te["end"].clip(upper=end)
    te = te.sort_values(["plot_start", "plot_end"])

    blocks: List[tuple[int, int]] = []
    for row in te[["plot_start", "plot_end"]].itertuples(index=False):
        block_start, block_end = int(row.plot_start), int(row.plot_end)
        if not blocks or block_start > blocks[-1][1]:
            blocks.append((block_start, block_end))
        else:
            blocks[-1] = (blocks[-1][0], max(blocks[-1][1], block_end))
    return te, blocks


def build_binned_profile(
    calls: pd.DataFrame,
    wgbs: pd.DataFrame,
    bin_table: pd.DataFrame,
    te_blocks: List[tuple[int, int]],
) -> pd.DataFrame:
    profile = bin_table.copy()
    edges = np.append(profile["bin_start"].to_numpy(), profile["bin_end"].iloc[-1])

    calls = calls.copy()
    calls["bin"] = pd.cut(
        calls["pos"], bins=edges, labels=False, include_lowest=True, right=False
    )
    ont = calls.groupby("bin")["prob1"].agg([("ont_mean", "mean"), ("ont_calls", "size")])
    profile = profile.merge(ont, left_on="bin", right_index=True, how="left")
    profile["ont_calls"] = profile["ont_calls"].fillna(0).astype(int)

    if wgbs.empty:
        profile["wgbs_mean"] = np.nan
        profile["wgbs_sites"] = 0
    else:
        wgbs = wgbs.copy()
        wgbs["bin"] = pd.cut(
            wgbs["pos"], bins=edges, labels=False, include_lowest=True, right=False
        )
        summary = wgbs.groupby("bin")["beta"].agg(
            [("wgbs_mean", "mean"), ("wgbs_sites", "size")]
        )
        profile = profile.merge(summary, left_on="bin", right_index=True, how="left")
        profile["wgbs_sites"] = profile["wgbs_sites"].fillna(0).astype(int)

    profile["is_te"] = [
        any(start < row.bin_end and end > row.bin_start for start, end in te_blocks)
        for row in profile.itertuples(index=False)
    ]
    return profile


def export_region_outputs(
    region: Dict[str, object],
    selected: pd.DataFrame,
    selected_stats: pd.DataFrame,
    matrix: pd.DataFrame,
    profile: pd.DataFrame,
    te: pd.DataFrame,
    output_dir: Path,
) -> None:
    region_id = str(region["region_id"])
    region_dir = output_dir / region_id
    region_dir.mkdir(parents=True, exist_ok=True)

    selected.sort_values(["read_id", "pos"]).to_csv(
        region_dir / "selected_per_read_calls.tsv.gz", sep="\t", index=False
    )
    selected_stats.to_csv(region_dir / "selected_read_summary.tsv", sep="\t", index=False)

    matrix_out = matrix.copy()
    matrix_out.columns = [f"bin_{value:03d}" for value in matrix_out.columns]
    matrix_out.to_csv(
        region_dir / "read_by_bin_probability_matrix.tsv.gz", sep="\t", na_rep="NA"
    )
    profile.to_csv(region_dir / "binned_profiles.tsv", sep="\t", index=False)
    te.to_csv(region_dir / "overlapping_te_intervals.tsv", sep="\t", index=False)


def plot_figure(
    results: List[Dict[str, object]], output_dir: Path, n_bins: int, max_reads: int
) -> None:
    fig = plt.figure(figsize=(8.9, 5.5), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1, 1, 0.62],
        height_ratios=[5.4, 1.7, 1.2],
        wspace=0.34,
        hspace=0.18,
    )
    cmap = LinearSegmentedColormap.from_list(
        "meth", ["#f7fbff", "#9ecae1", "#2b8cbe", "#084081"]
    )
    cmap.set_bad("#f1f3f5")
    quant_rows: List[Dict[str, object]] = []
    image = None

    for col, result in enumerate(results):
        region = result["region"]
        matrix = result["matrix"]
        profile = result["profile"]
        te = result["te"]
        te_blocks = result["te_blocks"]
        start = int(region["start"])
        end = int(region["end"])

        ax_heatmap = fig.add_subplot(gs[0, col])
        ax_line = fig.add_subplot(gs[1, col], sharex=ax_heatmap)
        ax_te = fig.add_subplot(gs[2, col], sharex=ax_heatmap)

        image = ax_heatmap.imshow(
            matrix.to_numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=[start, end, 0, matrix.shape[0]],
            origin="lower",
        )
        panel = "a" if col == 0 else "b"
        ax_heatmap.set_title(
            f"{panel}  {region['species_label']} {region['chrom']}:"
            f"{start / 1e6:.2f}-{end / 1e6:.2f} Mb",
            loc="left",
            fontsize=8.7,
            fontweight="bold",
            pad=10,
        )
        ax_heatmap.set_ylabel("Displayed reads" if col == 0 else "")
        ax_heatmap.set_yticks([0, matrix.shape[0]])
        ax_heatmap.set_yticklabels(["0", str(matrix.shape[0])])
        ax_heatmap.tick_params(axis="x", bottom=False, labelbottom=False)
        for block_start, block_end in te_blocks:
            ax_heatmap.axvspan(block_start, block_end, color="#5f6368", alpha=0.12, lw=0)

        ax_line.plot(
            profile["bin_center"],
            profile["ont_mean"],
            color="#1f78b4",
            lw=1.4,
            label="Nanopore read mean",
        )
        ax_line.plot(
            profile["bin_center"],
            profile["wgbs_mean"],
            color="#e76f51",
            lw=1.2,
            alpha=0.9,
            label="WGBS mean",
        )
        for block_start, block_end in te_blocks:
            ax_line.axvspan(block_start, block_end, color="#5f6368", alpha=0.12, lw=0)
        ax_line.set_ylim(0, 1)
        ax_line.set_ylabel("Mean methylation" if col == 0 else "")
        ax_line.spines["bottom"].set_visible(False)
        ax_line.tick_params(axis="x", bottom=False, labelbottom=False)
        if col == 0:
            ax_line.legend(loc="upper left", fontsize=7, handlelength=2.4)

        ax_te.set_ylim(0, 1)
        ax_te.set_yticks([])
        ax_te.set_ylabel("TEs" if col == 0 else "")
        ax_te.set_xlabel(f"{region['chrom']} position (Mb)")
        for row in te.itertuples(index=False):
            ax_te.add_patch(
                Rectangle(
                    (row.plot_start, 0.18),
                    row.plot_end - row.plot_start,
                    0.64,
                    facecolor="#6e6e6e",
                    edgecolor="none",
                    alpha=0.95,
                )
            )
        ax_te.set_xticks(np.linspace(start, end, 4))
        ax_te.set_xticklabels([f"{value / 1e6:.2f}" for value in np.linspace(start, end, 4)])
        ax_te.spines["left"].set_visible(False)
        ax_te.spines["right"].set_visible(False)
        ax_te.spines["top"].set_visible(False)

        for is_te, group in profile.groupby("is_te"):
            quant_rows.append(
                {
                    "region_id": region["region_id"],
                    "species": region["species_label"],
                    "context": "TE" if is_te else "non-TE",
                    "mean_ont_probability": float(group["ont_mean"].mean()),
                    "mean_wgbs_frequency": float(group["wgbs_mean"].mean()),
                    "n_bins": int(len(group)),
                    "n_bins_with_ont": int(group["ont_mean"].notna().sum()),
                    "n_bins_with_wgbs": int(group["wgbs_mean"].notna().sum()),
                }
            )

    quant = pd.DataFrame(quant_rows)
    quant.to_csv(output_dir / "te_non_te_bin_summary.tsv", sep="\t", index=False)

    ax_quant = fig.add_subplot(gs[:, 2])
    x = np.arange(len(results))
    width = 0.38
    non_te = quant[quant["context"] == "non-TE"]["mean_ont_probability"].to_numpy()
    within_te = quant[quant["context"] == "TE"]["mean_ont_probability"].to_numpy()
    labels = quant[quant["context"] == "TE"]["species"].tolist()
    ax_quant.bar(x - width / 2, non_te, width=width, color="#cfd8e3", label="non-TE")
    ax_quant.bar(x + width / 2, within_te, width=width, color="#4f7cac", label="TE")
    ax_quant.set_title(
        "c  Higher methylation within TE regions",
        loc="left",
        fontsize=8.5,
        fontweight="bold",
        pad=10,
    )
    ax_quant.set_ylabel("Mean read-level\nmethylation probability")
    ax_quant.set_ylim(0, 0.7)
    ax_quant.set_xticks(x)
    ax_quant.set_xticklabels(labels)
    ax_quant.legend(loc="upper left", fontsize=7.5, handlelength=1.2)
    for i, (outside, inside) in enumerate(zip(non_te, within_te)):
        ax_quant.text(
            i,
            max(outside, inside) + 0.03,
            f"{inside - outside:+.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#3f4c5a",
        )

    if image is not None:
        color_axis = fig.add_axes([0.925, 0.59, 0.018, 0.23])
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.set_label("Read-level methylation probability")

    base = output_dir / "fig_single_molecule_te_regions"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_bins <= 0 or args.max_reads <= 0:
        raise ValueError("--n-bins and --max-reads must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    regions = load_config(args.config, args.data_root)
    if len(regions) != 2:
        raise ValueError("The publication layout expects exactly two configured regions")
    validate_inputs(regions)

    results: List[Dict[str, object]] = []
    profile_rows: List[pd.DataFrame] = []
    for region in regions:
        calls = extract_region_calls(region, cache_dir, args.chunksize)
        selected, selected_stats = select_reads(calls, args.max_reads)
        matrix, bin_table = build_read_bin_matrix(
            selected, int(region["start"]), int(region["end"]), args.n_bins
        )
        wgbs = load_wgbs_region(
            [Path(path) for path in region["wgbs_beds"]],
            str(region["chrom"]),
            int(region["start"]),
            int(region["end"]),
            args.wgbs_min_coverage,
        )
        te, te_blocks = load_te_region(
            Path(region["te_bed"]),
            str(region["chrom"]),
            int(region["start"]),
            int(region["end"]),
        )
        profile = build_binned_profile(calls, wgbs, bin_table, te_blocks)
        profile.insert(0, "region_id", region["region_id"])
        profile_rows.append(profile)
        export_region_outputs(
            region, selected, selected_stats, matrix, profile, te, args.output_dir
        )
        results.append(
            {
                "region": region,
                "matrix": matrix,
                "profile": profile,
                "te": te,
                "te_blocks": te_blocks,
            }
        )
        print(
            f"{region['region_id']}: {calls['read_id'].nunique()} reads, "
            f"{calls['pos'].nunique()} cytosine positions, {len(te)} TE intervals"
        )

    pd.concat(profile_rows, ignore_index=True).to_csv(
        args.output_dir / "binned_profiles_all_regions.tsv", sep="\t", index=False
    )
    run_parameters = pd.DataFrame(
        [
            {
                "n_bins": args.n_bins,
                "max_reads": args.max_reads,
                "wgbs_min_coverage": args.wgbs_min_coverage,
                "missing_value": "NA",
                "read_selection": "n_calls_desc_then_read_id_asc",
                "read_order": "mean_prob1_desc_then_read_id_asc",
                "te_bin_rule": "any_overlap_with_merged_te_interval",
                "te_summary_unit": "genomic_bin",
            }
        ]
    )
    run_parameters.to_csv(args.output_dir / "run_parameters.tsv", sep="\t", index=False)
    plot_figure(results, args.output_dir, args.n_bins, args.max_reads)
    print(args.output_dir / "fig_single_molecule_te_regions.svg")


if __name__ == "__main__":
    main()
