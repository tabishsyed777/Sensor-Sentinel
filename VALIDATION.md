# Validation

- Tests were written first and initially failed because the implementation module did not exist.
- After implementation, all **14 unittest cases passed**.
- The 7,200-reading CLI demo completed and produced report JSON, prediction CSV, PNG, and SVG artifacts.
- The generated PNG was visually inspected: sensor traces, labeled fault spans, threshold, and alert markers render correctly.
- Model and baseline results in `examples/demo-report.json` are measured synthetic-demo outcomes, including the baseline's stronger point-level performance.
- No real machine data, production monitoring, or live deployment was evaluated.
- GitHub CI repeats tests and demo; check the Actions tab for the current run status.
