# STeep

STeep is a set of tools for distilling spatially-resolved transcriptomics (SRT) datasets by removing redundant spots/cells, accelerating downstream applications such as foundation model training and database queries.

# Contribution guidelines

<details>
<summary>Click to expand dev instructions!</summary>

## Dev environment setup

```bash
git clone https://github.com/alam-shahul/steep.git
cd steep
pip install --group=dev -e .
pre-commit install
```

Among other tools, the above installs the `pre-commit` command line tool. Once `pre-commit` is
active, every time you commit some changes, it will perform several code-style checks and
automatically apply some fixes for you (if there are any issues). When auto-fixes
are applied, you need to recommit those changes. Note that this process can require several
iterations.

After you are done committing changes and are ready to push the commits to the
remote branch, run `nox` to perform a final quality check. Note that `nox` is
linting only and does not fix the issues for you. You need to address the issues manually
based on the instructions provided.

## Writing unittests

We use [pytest](https://docs.pytest.org/en/stable/getting-started.html) to write unittests.
Unittests should always be added for new functionality. New test suites can be added under
`tests/test_{suite_name}.py`.

Run a particular test suite with:

```bash
python -m pytest tests/test_{suite_name}.py
```

Run all tests but the integration test:

```bash
python -m pytest -m 'not integration'
```

Note: to run the integration test, you'll need to specify the Hydra user using a `.env` file.
The contents of the file should be like so:

```bash
HYDRA_USER=test
```

## Code style and Nox

We adhere closely to the
[Google Python styleguide](https://google.github.io/styleguide/pyguide.html) for guidance on
standards of code style; please peruse the guide before contributing any code.

Nox is a tool that automates testing in multiple Python environments. Nox can be used to run
tests locally as well as remotely, e.g. on GitHub servers when committing code. Please invoke
Nox manually as a final quality check to maintain code style.

Run code linting and unittests:

```bash
nox
```

Run code linting only:

```bash
nox -e lint
```
