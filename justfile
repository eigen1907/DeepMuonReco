set shell := ["bash", "-c"]

# help
default: help

alias h := help
help:
    just -l -f {{justfile()}}

root_dir := justfile_directory()

commit:
    cat {{root_dir}}/.prompts/git-commit.md | copilot --allow-tool 'shell(git:*)' --deny-tool 'shell(git push)'

sanity_check:
    uv run python {{root_dir}}/train.py debug=sanity-check
