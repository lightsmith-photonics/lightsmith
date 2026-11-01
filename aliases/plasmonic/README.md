# plasmonic → Lightsmith

**`plasmonic` is now [Lightsmith](https://pypi.org/project/lightsmith/)**, the open photonics suite: one typed
JSON problem spec solved by Mie, T-matrix, DDA, FDTD, TMM, RCWA, FDE, EME, FDFD, PWE and circuit solvers on
CPU or GPU, with differentiable inverse design, for humans and AI agents.

This package only exists so that the name leads you there: installing it installs `lightsmith`, and
`import plasmonic` gives you the `lightsmith` API. Please depend on `lightsmith` directly:

```bash
pip install lightsmith            # or: uv add lightsmith
pip install "lightsmith[gpu]"     # + PyTorch for the GPU solvers
```

```python
import lightsmith as ls
```

- Documentation: <https://docs.lightsmith.dev/>
- Source: <https://github.com/lightsmith-photonics/lightsmith>

MIT licence.
