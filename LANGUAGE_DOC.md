# NeuroScript DSL Documentation

## Overview

NeuroScript is a domain-specific language (DSL) designed for describing, simulating, and visualizing neural networks—both traditional Artificial Neural Networks (ANN) and spiking/biological Neural Networks (SNN). The DSL is transpiled into Python code using PyTorch (for ANN) and Brian2 (for SNN). It supports various training methods, model save/load, data input/output, and on‑the‑fly code interpretation.

## Syntax

### 1. Network Definition
Define a network with a name and type:
```neuroscript
network MyNetwork type ANN {
    // Network body goes here
}
```
Supported types: `ANN`, `SNN`.

### 2. Layers
For ANN:
```neuroscript
input_layer {
    neurons: 784;
}

hidden_layer {
    neurons: 128;
    activation: relu;
}

output_layer {
    neurons: 10;
    activation: softmax;
}
```
For SNN (using generic layers):
```neuroscript
layer LayerName {
    neurons: 100;
    type: Excitatory;
}
```

### 3. Connections
Define connections between layers:
```neuroscript
connect input_layer -> hidden_layer {
    weight_init: xavier;
}
```

### 4. Training Parameters
Specify training settings including optimizer, learning rate, loss function, epochs, batch size, and optional scheduler parameters:
```neuroscript
training {
    optimizer: SGD;
    lr: 0.01;
    momentum: 0.9;
    loss: cross_entropy;
    epochs: 10;
    batch_size: 32;
    scheduler: StepLR;
    step_size: 5;
    gamma: 0.5;
}
```

### 5. Data Input
Define data sources for training or prediction:
```neuroscript
data {
    train: "train_data.csv";
}
```

### 6. Model Save/Load and Prediction
Commands for saving, loading, and predicting:
```neuroscript
save_model "mymodel.pth";
load_model "mymodel.pth";
predict "predict_input.csv";
```

### 7. Mathematical Functions
NeuroScript supports vector and matrix literals.

**Vector literal:**
```neuroscript
vector [1, 2, 3.5, 4]
```

**Matrix literal:**
```neuroscript
matrix [[1, 2], [3, 4]]
```
These literals can be used for specifying weight initializations or other mathematical parameters.

### 8. Simulation and Visualization
Execute simulation and visualization commands:
```neuroscript
simulate MyNetwork;
visualize MyNetwork;
```

## Execution Modes

- **CLI Mode:**  
  Run a DSL file:
  ```bash
  python main.py examples/ann_example.ns
  ```
  Or pass DSL code as a string:
  ```bash
  python main.py "network MyNetwork type ANN { ... }" -s
  ```
  Optionally, specify an output filename:
  ```bash
  python main.py examples/ann_example.ns -o MyNetwork.py
  ```

- **Python API (On-the-Fly Interpretation):**  
  Import and call the interpreter:
  ```python
  from interpreter import interpret_neuroscript
  
  dsl_code = '''
  network MyNetwork type ANN {
      // DSL code here...
  }
  simulate MyNetwork;
  visualize MyNetwork;
  '''
  interpret_neuroscript(dsl_code)
  ```

## Training Methods

NeuroScript supports multiple training methods:
- **Optimizers:** Choose from Adam, SGD, etc.
- **Learning Rate:** Set with the `lr` parameter.
- **Schedulers:** For example, StepLR (with `step_size` and `gamma`).
- **Additional Options:** Such as momentum for SGD.

The generated Python code will include a training loop that iterates through epochs and batches as defined in the DSL.

## Example

```neuroscript
network MyNetwork type ANN {
    input_layer {
        neurons: 784;
    }
    hidden_layer {
        neurons: 128;
        activation: relu;
    }
    output_layer {
        neurons: 10;
        activation: softmax;
    }
    connect input_layer -> hidden_layer {
        weight_init: xavier;
    }
    connect hidden_layer -> output_layer {
        weight_init: xavier;
    }
    training {
        optimizer: SGD;
        lr: 0.01;
        momentum: 0.9;
        loss: cross_entropy;
        epochs: 10;
        batch_size: 32;
        scheduler: StepLR;
        step_size: 5;
        gamma: 0.5;
    }
    data {
        train: "train_data.csv";
    }
    save_model "mymodel.pth";
    predict "predict_input.csv";
}
simulate MyNetwork;
visualize MyNetwork;
```

## Conclusion

NeuroScript provides a high-level, human-readable way to define neural network architectures, training procedures, and data operations, which are automatically converted into executable Python code. For further details and examples, please refer to this documentation and the source code in the repository.
