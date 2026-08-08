#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_file="$root_dir/results/paper_figures/oatm_architecture_pipeline.dot"
output_dir="$root_dir/results/paper_figures"
stem="$output_dir/oatm_architecture_pipeline"

mkdir -p "$output_dir"

dot -Tsvg "$source_file" -o "$stem.svg"
dot -Tpdf "$source_file" -o "$stem.pdf"
dot -Tpng -Gdpi=240 "$source_file" -o "$stem.png"

printf 'Wrote:\n%s\n%s\n%s\n' "$stem.svg" "$stem.pdf" "$stem.png"
