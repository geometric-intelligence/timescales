# Timescales in Recurrent Neural Networks


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


## To set up the kernel for Jupyter notebooks:

1. Run the following command:

```bash
python -m ipykernel install --user --name timescales --display-name "Python (timescales)"
```

2. Verify the kernel was created:

```bash
jupyter kernelspec list
```

3. Refresh the kernel selector. If it doesn't show up, then (if using VSCode/Cursor): press `Command+Shift+P` to open the command palette, and select `Developer: Reload Window` to refresh the window.
