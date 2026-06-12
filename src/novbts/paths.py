"""Central path config — resolve project locations from the repo root so scripts
work regardless of the current working directory (replaces hardcoded "data/..."
strings and sys.path hacks).

Layout:
    ROOT/
      data/analytic/   (analytic Hertz-Mindlin train/test/ood)
      data/fem/        (PhysX-FEM ground truth: normal.npz, shear_*.npz, chunks/)
      runs/            (training/eval outputs)
      docs/            (reports)
"""
from pathlib import Path

# src/novbts/paths.py -> parents[2] == repo root
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
ANALYTIC = DATA / "analytic"
FEM = DATA / "fem"
RUNS = ROOT / "runs"
DOCS = ROOT / "docs"
LOGS = ROOT / "logs"


def ensure(*dirs: Path) -> None:
    """mkdir -p for any number of dirs (call before writing outputs)."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
