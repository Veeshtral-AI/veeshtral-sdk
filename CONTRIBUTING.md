# Contributing

## Branch & PR policy (required)

**Do not push commits directly to `main`.**

All changes go through a short-lived branch and a pull request:

1. Branch from latest `main`:
   ```bash
   git fetch origin
   git checkout main
   git pull origin main
   git checkout -b feat/<short-name>   # or fix/…, chore/…, docs/…
   ```
2. Commit on the branch (small, focused commits).
3. Push and open a PR into `main`:
   ```bash
   git push -u origin HEAD
   gh pr create --base main --title "…" --body "…"
   ```
4. Wait for CI (`pytest` + coverage ≥ 80%).
5. Merge via GitHub (squash preferred for chores; merge commit OK for multi-commit features).
6. Delete the branch after merge.

### Why

`main` is the release surface for the `veeshtral` package (and future PyPI tags). PRs give review, CI gates, and a clear history before anything is tagged or published.

### Exceptions

- Initial bootstrap of an empty repo (already done).
- Emergency hotfix only with explicit maintainer approval — still prefer a PR if CI can run.

## Local development

```bash
pip install -e ".[dev]"
python -m pytest tests -c pytest.ini --cov=veeshtral --cov-config=.coveragerc --cov-fail-under=80
```

## Releasing to PyPI

Tracked in [#1](https://github.com/Veeshtral-AI/veeshtral-sdk/issues/1). Automation lives in [`.github/workflows/publish.yml`](.github/workflows/publish.yml) (Trusted Publishing / OIDC — **no API tokens in the repo**).

### One-time maintainer setup

1. Create GitHub Environments **`pypi`** and **`testpypi`** on this repo (require reviewers on `pypi`).
2. On [PyPI Trusted Publishers](https://pypi.org/manage/account/publishing/) (and TestPyPI), register a **pending** publisher:
   - PyPI project name: `veeshtral`
   - Owner: `Veeshtral-AI`
   - Repository: `veeshtral-sdk`
   - Workflow name: `publish.yml`
   - Environment name: `pypi` (or `testpypi`)
3. First successful publish claims the project name.

### Release checklist

1. On a feature branch, bump **both**:
   - `pyproject.toml` → `[project].version`
   - `src/veeshtral/__init__.py` → `__version__`
2. Open PR → wait for CI → merge to `main`.
3. Tag and push (tag must match the version, e.g. `0.1.4` → `v0.1.4`):
   ```bash
   git checkout main && git pull
   git tag -a v0.1.4 -m "veeshtral 0.1.4"
   git push origin v0.1.4
   ```
4. Approve the `pypi` environment deployment if required.
5. Confirm: https://pypi.org/project/veeshtral/ and `pip install veeshtral==0.1.4`

### TestPyPI dry-run (no tag)

Actions → **Publish** → Run workflow → target `testpypi`.

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ veeshtral
```

### Manual build (optional)

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
# Do not upload with long-lived tokens in CI; prefer the Publish workflow.
```

**Never re-upload an existing version.** Bump first.

## Related issues

- [#1](https://github.com/Veeshtral-AI/veeshtral-sdk/issues/1) — PyPI publish
- [#2](https://github.com/Veeshtral-AI/veeshtral-sdk/issues/2) — LangGraph adapter (v1.1)
