# Code for Project 1

This directory contains the code for Project 1. The implementation is found in the `scripts/` directory, while the code for generating the plots is located in the `tasks/` directory. The code for the second part of the project is located in the `stan/` directory.

## Installation

The easiest way to install the required dependencies is to use uv. Follow the instructions in [Installing uv](https://docs.astral.sh/uv/getting-started/installation/) to set up on your machine. Make sure uv is correctly installed by running

```bash
uv --version
```

If you want, you can also easily install the latest version of Python through uv by running (note that this is optional, but it ensures you have the correct version of Python for the project)

```bash
uv python install
```
Finally, generate a virtual environment and install packages specified in `pyproject.toml` by using the command

```bash
uv sync
```

You can now run the code in this project by using `uv run <command>`. F.ex. run the `tasks/rwmh_chain.py` script by using
```bash
uv run python tasks/rwmh_chain.py
```

Alternatively, you can use pip and venv to set up a virtual environment and install the dependencies. For this you install the packages in `pyproject.toml` by running

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -e . --group dev
```

## Generating all plots
Use the shell script `run_all_experiments.sh` to run all the code for generating the plots. This will run all the scripts in `tasks/`. You can run the script by using
```bash
sh run_all_experiments.sh
```


