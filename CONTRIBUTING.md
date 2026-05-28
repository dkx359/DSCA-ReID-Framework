# Contributing

Contributions are welcome if they improve the clarity, reliability, or extensibility of the DSCA research framework.

## Development setup

```bash
git clone https://github.com/dkx359/DSCA-ReID-Framework.git
cd DSCA-ReID-Framework
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## Local checks

Before opening a pull request, run:

```bash
pytest -q
python scripts/smoke_check.py --config configs/lightweight.yaml --out outputs/smoke
```

If you use Ruff locally, also run:

```bash
ruff check src scripts tests
```

## Pull request guidelines

- Keep the lightweight backend runnable without external checkpoints.
- Do not commit datasets, pretrained weights, generated outputs, or private data.
- Document new replacement points for real backends in `docs/REPLACEMENT_GUIDE.md`.
- Add or update tests when changing model, sampler, loss, metric, or config behavior.
- Clearly distinguish lightweight sanity-check results from paper-level reproduction results.

## Preferred contribution areas

- Real backend wrappers for OpenPose, DeepLabV3, CLIP, Stable Diffusion, and ReID surrogates.
- More faithful evaluation protocols for Market-1501, DukeMTMC-reID, MSMT17, and PRCC.
- Additional unit tests for schedulers, masks, metrics, and checkpoint resume behavior.
- Documentation improvements that make the framework easier to extend responsibly.
