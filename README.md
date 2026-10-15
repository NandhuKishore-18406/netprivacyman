# Net-track: Generic Web Tracking Detection Using Packet Metadata
## Faithful Reproduction Project (WWW 2023)

A standalone reproduction project implementing the Random Forest classifier methodology from:
> **Hyung-Kyu Lee, Joo-Young Joo, and Dong-Hoon Lee.**  
> *"Net-track: Generic Web Tracking Detection Using Packet Metadata."*  
> Proceedings of the ACM Web Conference 2023 (WWW '23).

---

### Key Highlights
- **Scope:** Random Forest classifier & Algorithm 1 feature extraction.
- **Excluded Classifiers:** Decision Tree, k-NN, MLP, and LSTM were intentionally excluded per the paper's own empirical finding (Section 5.1) establishing Random Forest as the best-performing, most practical choice.
- **Dependency Management:** Python 3.12 with `uv` package manager (`pyproject.toml`).
- **Architecture:** 6 core files in `src/netrack/`.

---

### Detailed Technical Explanation
For a file-by-file technical breakdown explaining what each module does and how it works under the hood, see:
- [PROJECT_EXPLANATION.md](file:///home/nandhu/projects/mlproject%28netrack%29/PROJECT_EXPLANATION.md)

---

### Quick Start

```bash
# Activate Virtual Environment (Fish shell)
source .venv/bin/activate.fish

# Run Unit Tests
uv run python -m unittest discover tests

# Execute Full Reproduction Pipeline
uv run python -m netrack.cli pipeline
```
