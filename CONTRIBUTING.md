## Contributing to OmniSafe

If you are interested in contributing to OmniSafe, your contributions will fall
into two categories:

1. You want to propose a new Feature and implement it
    - Create an issue about your intended feature, and we shall discuss the design and
    implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue
    - Look at the outstanding issues here: https://github.com/OmniSafeAI/omnisafe/issues
    - Pick an issue or feature and comment on the task that you want to work on this feature.
    - If you need more context on a particular issue, please ask and we shall provide.

Once you finish implementing a feature or bug-fix, please send a Pull Request to https://github.com/OmniSafeAI/omnisafe

If you are not familiar with creating a Pull Request, here are some guides:

- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/

## Developing OmniSafe

To develop OmniSafe on your machine, here are some tips:

1. Clone a copy of OmniSafe from GitHub:

```bash
git clone https://github.com/OmniSafeAI/omnisafe
cd omnisafe/
```

2. Install OmniSafe in develop mode, with support for building the docs and running tests:

```bash
pip install -e .[docs,tests,extra]
```

## Codestyle

We are using [black codestyle](https://github.com/psf/black) (max line length of 127 characters) together with [isort](https://github.com/timothycrosley/isort) to sort the imports.

**Please run `make format`** to reformat your code. You can check the codestyle using `make check-codestyle` and `make lint`.

Please document each function/method and [type](https://google.github.io/pytype/user_guide.html) them using the following template:

```python
def my_function(arg1: type1, arg2: type2) -> returntype:
    """
    Short description of the function.

    :param arg1: describe what is arg1
    :param arg2: describe what is arg2
    :return: describe what is returned
    """
    ...
    return my_variable
```

## Pull Request (PR)

Before proposing a PR, please open an issue, where the feature will be discussed. This prevent from duplicated PR to be proposed and also ease the code review process.

Each PR need to be reviewed and accepted by at least one of the maintainers (Borong Zhang, [Jiayi Zhou](https://github.com/Gaiejj), [JTao Dai](https://github.com/calico-1226), [Weidong Huang](https://github.com/hdadong), [Xuehai Pan](https://github.com/XuehaiPan) and [Jiamg Ji](https://github.com/zmsn-2077)).
A PR must pass the Continuous Integration tests to be merged with the master branch.

## Tests

All new features must add tests in the `tests/` folder ensuring that everything works fine.
We use [pytest](https://pytest.org/).
Also, when a bug fix is proposed, tests should be added to avoid regression.

To run tests with `pytest`:

```bash
make pytest
```

Type checking with `pytype`:

```bash
make type
```

Codestyle check with `black`, `isort` and `flake8`:

```bash
make check-codestyle
make lint
```

To run `pytype`, `format` and `lint` in one command:

```bash
make commit-checks
```

Build the documentation:

```bash
make docs
```

Check documentation spelling (you need to install `sphinxcontrib.spelling` package for that):

```bash
make spelling
```

## Changelog and Documentation

Please do not forget to update the changelog (`docs/misc/changelog.rst`) and add documentation if needed.
You should add your username next to each changelog entry that you added. If this is your first contribution, please add your username at the bottom too.
A README is present in the `docs/` folder for instructions on how to build the documentation.

Credits: this contributing guide is based on the [PyTorch](https://github.com/pytorch/pytorch/) one.
