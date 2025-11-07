# Multi-timescale Recurrent Neural Networks

<h3 align="center">
    Multi-timescale Recurrent Neural Networks trained on neuroscience tasks:
</h3>


<div align="center">
    <img src="assets/pirnns.jpg" width="600">
    <p><em>Source: <a href="https://www.sciencedirect.com/science/article/pii/S0960982223000659">ScienceDirect Article</a></em></p>
</div>


<h3 align="center">
    Multi-timescale Recurrent Neural Networks:
</h3>


<div align="center">
    <img src="assets/overview.jpg" width="600">
</div>

## Getting Started

### Clone the repository & navigate to the directory:

```bash
git clone https://github.com/geometric-intelligence/timescales.git
cd timescales
```

### Install Poetry

This project uses [Poetry](https://python-poetry.org/) to manage dependencies.

1. Install Poetry (if you don't have it already):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Make sure that Poetry is accessible in your PATH. For example, add the following to your `.zshrc` file:

```bash
export PATH="~/.local/bin:$PATH"
```

3. Check that Poetry is installed correctly:

```bash
poetry --version
```

### Setup the environment

1. Install dependencies

```bash
poetry install
```

2. Install the poetry shell plugin

```bash
poetry self add poetry-plugin-shell
```

3. Activate the virtual environment

```bash
poetry shell
```

### Start coding!

```bash
python my_script.py
```

## Before sending a PR, make sure to format, lint, type check, and test the code:

```bash
black .
ruff check --fix .
mypy .
pytest .
```


