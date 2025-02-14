# NeuroScript

**NeuroScript** is a domain-specific language (DSL) for describing, simulating, and visualizing neural networks (both traditional and spiking/biological). The project automatically transpiles DSL code into Python code using [PyTorch](https://pytorch.org/) for ANN and [Brian2](https://brian2.readthedocs.io/) for SNN.

## Features

- **DSL Execution via CLI or Python API:**  
  Provide a path to a DSL file or a DSL code string.

- **Model Training:**  
  Generate training loops using training parameters and data specified in the DSL.

- **Model Saving and Loading:**  
  Use `save_model` and `load_model` commands in DSL to persist or load models.

- **Data Input/Output:**  
  Specify data sources for training and prediction via the `data` and `predict` blocks.

## Project Structure
```
nspy/
    ├── parser/ # Language parser (lexer, AST, syntax analysis) 
    ├── transpiler/ # Code generation modules (ANN and SNN) 
    ├── visualizer/ # Network visualization module (static and/or dynamic) 
    ├── examples/ # Example DSL files (.ns) 
    ├── main.py # Command-line interface (CLI) entry point and Python API 
    ├── README.md # Project documentation 
    └── .gitignore # Git ignore file
```

## Installation

1. Ensure you have [Python 3](https://www.python.org/) installed.

2. Install the required dependencies:
```bash
    pip install ply torch brian2 networkx matplotlib pandas
```
## Usage
### Via CLI

Run the project from the command line by specifying the DSL source file:

```bash
python main.py examples/ann_example.ns
```

For an SNN example:

```bash
python main.py examples/snn_example.ns
```

Optionally, specify an output file name:

```bash
python main.py examples/ann_example.ns -o MyNetwork.py
```

If you want to pass DSL code as a string:

```bash
python main.py "network MyNetwork type ANN { ... }" -s
```

### Via Python API

You can also import and use the DSL processing functionality in your own Python scripts by calling functions from the parser and transpiler modules.

## Development
- Parser: Located in the parser/ directory.
- Code Generation: Modules ann_generator.py and snn_generator.py in the transpiler/ directory.
- Visualization: Located in the visualizer/ directory.
- CLI and Python API: Entry point is main.py.
## License

This project is licensed under the MIT License.

## Authors

Dmitry Manushin