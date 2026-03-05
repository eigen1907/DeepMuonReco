#!/usr/bin/env fish
set -xg PROJECT_ROOT (dirname (readlink -m (status --current-filename)))
source $PROJECT_ROOT/.venv/bin/activate.fish
