# Releasing lightsmith

How the owner publishes a release of the `lightsmith` library to PyPI, mints a Zenodo DOI and publishes
the documentation site. Every command runs from a clean checkout of `main` in this repository.
Nothing here runs in CI: the project has no GitHub Actions.

The PyPI name `lightsmith` was free on 2026-11-01. The first upload claims it for the account that
uploads.

## 1. Prepare the release

1. Make sure `main` contains everything for the release and the working tree is clean:

   ```bash
   git switch main && git pull && git status
   ```

2. Set the version in **both** places:
   - `version` in `pyproject.toml`
   - `__version__` in `src/lightsmith/__init__.py`

   Also update `version:` in `CITATION.cff`, and add `date-released: YYYY-MM-DD` there.
3. In `CHANGELOG.md`, replace `unreleased` in the release heading with today's date, for example
   `## [0.1.0] - 2026-11-15`, and keep an empty `## [Unreleased]` section above it.
4. Run the tests and the docs build:

   ```bash
   uv sync --extra gpu --extra mcp
   uv run pytest
   uv run --group docs mkdocs build --strict
   ```

5. Commit the version bump: `git commit -am "Release 0.1.0"`.

## 2. Build and check the artifacts

```bash
rm -rf dist
uv build
uvx twine check --strict dist/*
```

`uv build` writes `dist/lightsmith-<version>.tar.gz` (sdist) and `dist/lightsmith-<version>-py3-none-any.whl`.
The wheel contains the package, the bundled material data (`lightsmith/data/materials/*`) and `py.typed`.
The sdist also contains `tests/`, `examples/` and `docs/`; it leaves out `benchmarks/` (results are large)
and `scripts/`.

The `[tool.uv.sources]` PyTorch CUDA index only affects `uv sync` in this repository. It is not part of
the published metadata. To confirm, check that the wheel's METADATA has a plain `torch>=2.7; extra == 'gpu'`
and no index URL:

```bash
unzip -p dist/lightsmith-*.whl 'lightsmith-*.dist-info/METADATA' | grep -i -E "torch|index"
```

Then smoke-test the wheel in a clean virtual environment, outside the repository, so that the
installed package is the one under test:

```bash
cd "$(mktemp -d)"
uv venv --python 3.12
uv pip install "/path/to/lightsmith/dist/lightsmith-0.1.0-py3-none-any.whl[mcp]"
.venv/bin/lightsmith doctor          # on Windows: .venv/Scripts/lightsmith.exe
.venv/bin/python -I -c "
import lightsmith as ls, lightsmith.mcp_server as m
sim = ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=40), material=ls.LibraryMaterial(name='Au'))],
                    medium_index=1.33, wavelengths=ls.WavelengthRange(start_nm=400, stop_nm=800, num=41))
print(ls.solve(sim).method, m.build_server())"
```

To test the upload itself first, use TestPyPI. It needs a separate account and token from
<https://test.pypi.org>:

```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token <TestPyPI token>
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ lightsmith
```

## 3. Publish to PyPI

1. In your PyPI account, go to **Account settings → API tokens** and create a token. For the first
   upload the scope must be "Entire account", because the project does not exist yet. After the first
   upload, replace it with a token scoped to the `lightsmith` project.
2. Upload. `uv publish` reads the token from `UV_PUBLISH_TOKEN`, or you pass `--token`. Do not store the
   token in the repository.

   ```bash
   export UV_PUBLISH_TOKEN=pypi-...      # PowerShell: $env:UV_PUBLISH_TOKEN = "pypi-..."
   uv publish
   ```

   With no token set, `uv publish` asks for a username and password: the username is `__token__` and the
   password is the token.
3. Check <https://pypi.org/project/lightsmith/>, then install from PyPI in a clean environment:
   `uv pip install lightsmith && lightsmith doctor`.

### Trusted publishing (later)

PyPI's [trusted publishing](https://docs.pypi.org/trusted-publishers/) replaces API tokens with an
OIDC identity from a CI provider such as GitHub Actions or GitLab CI. Because this project has no GitHub
Actions, releases use a token for now. To switch later:

1. Add a workflow that builds `dist/` and runs `uv publish` (uv picks up the OIDC token by itself).
2. Register that workflow as a trusted publisher on PyPI under **lightsmith → Settings → Publishing**.
3. Delete the API token.

## 4. Tag, GitHub release and Zenodo DOI

### One-time Zenodo setup (before the first release)

1. Sign in to <https://zenodo.org> with GitHub.
2. Grant Zenodo access to the `lightsmith-photonics` organisation:
   - In GitHub, open **Settings → Applications → Authorized OAuth Apps → Zenodo**.
   - Click **Grant** next to the organisation, or ask an org owner to approve the request.
3. On Zenodo, open **Account → GitHub**, click **Sync now**, and switch on `lightsmith-photonics/lightsmith`.

From then on, every *published GitHub release* of the repository is archived on Zenodo and gets a DOI.
Zenodo reads the record metadata from `.zenodo.json`; `CITATION.cff` is what GitHub's "Cite this
repository" button shows. Keep the title, authors and licence the same in both files.

### Every release

```bash
git tag -a v0.1.0 -m "lightsmith 0.1.0"
git push origin main v0.1.0
gh release create v0.1.0 dist/* --title "lightsmith 0.1.0" --notes-file <(sed -n '/^## \[0.1.0\]/,/^## \[/p' CHANGELOG.md | sed '$d')
```

After a few minutes the release appears on Zenodo with two DOIs:

- a **version DOI**, for this release only;
- a **concept DOI**, which always resolves to the latest version.

After the first release, add the concept DOI to the citation files and the README, then commit:

- In `CITATION.cff`, add:

  ```yaml
  identifiers:
    - type: doi
      value: 10.5281/zenodo.NNNNNNN
      description: Concept DOI (all versions)
  ```

  or set `doi: 10.5281/zenodo.NNNNNNN`.
- In the README, add a DOI badge:

  ```
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.NNNNNNN.svg)](https://doi.org/10.5281/zenodo.NNNNNNN)
  ```

## 5. Documentation site

The site is MkDocs Material, built from `docs/` and `mkdocs.yml` and served by GitHub Pages from the
`gh-pages` branch at <https://lightsmith-photonics.github.io/lightsmith/>.

Preview locally:

```bash
uv run --group docs mkdocs serve
```

Publish:

```bash
uv run --group docs mkdocs gh-deploy --strict --force -m "Deploy docs for {sha}"
```

`gh-deploy` builds the site, commits it to the `gh-pages` branch and pushes that branch to `origin`.
The first time, enable Pages in GitHub:

1. Open **Settings → Pages** for `lightsmith-photonics/lightsmith`.
2. Set **Source** to *Deploy from a branch*, then choose branch `gh-pages`, folder `/ (root)`.

Re-run `gh-deploy` after each release, or whenever the docs on `main` change.

## 6. After the release

### Sibling repositories

The sibling repositories (`lightsmith-cloud`, `lightsmith-materials`, `lightsmith-studio`, `lightsmith-adda`,
`lightsmith-scattnlay`, ...) currently take the library from a sibling checkout through a uv path source:

```toml
[tool.uv.sources]
lightsmith = { path = "../lightsmith", editable = true }
```

Once `lightsmith` is on PyPI, switch each of them to the published package:

1. In the repository's `pyproject.toml`, delete the `lightsmith = { path = ... }` line from
   `[tool.uv.sources]`. Delete the whole table if that was its only entry, but keep any `torch` index pin.
2. Make sure the dependency has a lower bound, for example `"lightsmith>=0.1"`.
3. Run `uv lock` and `uv sync`, then run that repository's tests.

For local co-development against an unreleased library change, keep the path source on a branch, or
run `uv pip install -e ../lightsmith` in that repository's environment.

### Start the next cycle

Open the next cycle by adding new entries under `## [Unreleased]` in `CHANGELOG.md`. Bump the version in
`pyproject.toml` and `__init__.py` when you prepare the next release.
