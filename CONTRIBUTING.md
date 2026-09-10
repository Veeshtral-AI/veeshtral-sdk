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

## Related issues

- [#1](https://github.com/Veeshtral-AI/veeshtral-sdk/issues/1) — PyPI publish
- [#2](https://github.com/Veeshtral-AI/veeshtral-sdk/issues/2) — LangGraph adapter (v1.1)
