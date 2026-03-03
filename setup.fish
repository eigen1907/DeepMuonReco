#!/usr/bin/env fish
set -xg PROJECT_ROOT (dirname (readlink -m (status --current-filename)))
set -xga PYTHONPATH {$PROJECT_ROOT}/src
micromamba activate muonly-py312
