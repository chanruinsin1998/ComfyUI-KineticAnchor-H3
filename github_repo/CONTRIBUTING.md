# Contributing

Issues and pull requests are welcome. Reports may be written in English or Chinese.

## Bug reports

Open an issue using the bug report template. Include the ComfyUI version, the workflow JSON, and the console output. Reduced workflows that reproduce the problem are handled first.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep changes focused; unrelated refactoring belongs in a separate pull request.
3. Verify that all sources compile: `python -m compileall -q .`
4. Run the offline regression test where applicable: `python test_nodes.py`
5. Describe the motivation and the behavior change in the pull request.

## Code style

The codebase targets the Python environment bundled with ComfyUI. Follow the style of the surrounding code and do not reformat untouched lines.
