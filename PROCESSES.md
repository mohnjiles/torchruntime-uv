# Processes
## PCI Database Updates
A [GitHub Actions workflow](.github/workflows/pci-update-workflow.yml) runs automatically every Thursday at 3.14am UTC to fetch the latest database for PCI IDs from [here](https://raw.githubusercontent.com/pciutils/pciids/refs/heads/master/pci.ids).

If it finds a new database (by comparing the sha256 against the stored [sha256 sum](pci.ids.sha256)), it will create an SQLite database of graphics cards using [txt_to_db.py](scripts/txt_to_db.py). It will then update the stored .sha256 sum, increment the minor version in [pyproject.toml](pyproject.toml), and create a new GitHub release.

This new release will automatically trigger the PyPI publishing workflow (described next).

## PyPI Publishing
A [GitHub Actions workflow](.github/workflows/pypi-release-workflow.yml) will run automatically whenever a new release is created. This will publish a new release on PyPI using the version number in [pyproject.toml](pyproject.toml).

This can be triggered manually by creating a new release (after updating [pyproject.toml](pyproject.toml)), for e.g. to publish code changes. Or it will trigger automatically if the PCI database updates via the [PCI update workflow](.github/workflows/pci-update-workflow.yml).

Before publishing to PyPI, this workflow automatically tries to install the generated wheel (`.whl`), and runs `pytest` on it (the generated wheel contains all the tests in the `tests/` folder).

## Automated Tests
Every commit to the repo will run `pytest` with Python 3.8 and Python 3.11.

Before publishing to PyPI, the [PyPI workflow](.github/workflows/pypi-release-workflow.yml) automatically tries to install the generated wheel (`.whl`), and runs `pytest` on it (the generated wheel contains all the tests in the `tests/` folder).

Running `pytest` also runs integrated tests to verify the SQLite database bundled in `torchruntime`.
