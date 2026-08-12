# Per-read methylation across TE-rich regions

`plot_per_read_te_regions.py` reproduces the optional single-molecule analysis
described in Box 2 of the protocol. It reads the per-read `call_mods.tsv` output,
reference-matched TE BED files and optional WGBS BED files listed in
`fig5_te_regions.tsv`.

Create the lightweight plotting environment once:

```bash
conda env create -f scripts/single_molecule_te/environment_single_molecule_te.yml
```

Run the analysis without modifying the chemistry-specific compute environment:

```bash
conda run -n deepsignal-plant-te python scripts/single_molecule_te/plot_per_read_te_regions.py \
  --data-root /path/to/protocol_data \
  --config scripts/single_molecule_te/fig5_te_regions.tsv \
  --output-dir per_read_te_analysis \
  --max-reads 120 \
  --n-bins 220 \
  --wgbs-min-coverage 5
```

The script uses `prob1` as the methylation probability, retains the 120 reads
with the largest number of calls in each 50-kb interval, and orders them by
decreasing mean `prob1`. The 220 equal-width bins are approximately 227.3 bp.
Read-bin combinations without a call are written as `NA`. A bin is classified
as TE-associated when it overlaps any merged TE interval, and TE versus non-TE
means use genomic bins as the aggregation unit.

Outputs include the selected per-read records, selected-read summary,
read-by-bin matrix, binned Nanopore and WGBS profiles, overlapping TE intervals,
TE versus non-TE summary, run parameters and PNG, SVG, PDF and TIFF figures.
