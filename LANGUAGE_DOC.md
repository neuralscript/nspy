# NeuroScript DSL Documentation

## Overview

NeuroScript is a domain-specific language (DSL) designed for describing, simulating, and visualizing neural networks—both traditional artificial neural networks (ANN) and spiking neural networks (SNN). The DSL is transpiled into Python code using PyTorch (for ANN) and Brian2 (for SNN). It supports specifying network architectures, training configurations, data input, model saving/loading, and predictions.

---

## Syntax

### 1. Network Definition
Define a network with a name and type:
```neuroscript
network MyNetwork type ANN {
    // Network body goes here
}
```
Supported types: `ANN`, `SNN`.

---

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
For SNN (use generic layer):
```neuroscript
layer LayerName {
    neurons: 100;
    type: Excitatory;
}
```

---

### 3. Connections
Define connections between layers:
```neuroscript
connect input_layer -> hidden_layer {
    weight_init: xavier;
}
```

---

### 4. Training Parameters
Define training parameters including optimizer, loss, epochs, batch size, and additional training options:
```neuroscript
training {
    optimizer: Adam;
    lr: 0.001;
    loss: cross_entropy;
    epochs: 10;
    batch_size: 64;
    scheduler: StepLR;
    step_size: 5;
    gamma: 0.1;
}
```

---

### 5. Data Input
Specify data for training or prediction:
```neuroscript
data {
    train: "train_data.csv";
}
```

---

### 6. Model Save/Load and Prediction
```neuroscript
save_model "mymodel.pth";
load_model "mymodel.pth";
predict "predict_input.csv";
```

---

### 7. Simulation and Visualization
```neuroscript
simulate MyNetwork;
visualize MyNetwork;
```

---

## Execution Modes

### CLI Mode
Run a DSL file:
```bash
python main.py examples/ann_example.ns
```
Or pass code as a string:
```bash
python main.py "network MyNetwork type ANN { ... }" -s
```

### Python API Mode
Import the `interpret_neuroscript` function from `interpreter.py`:
```python
from interpreter import interpret_neuroscript

dsl_code = '''
network MyNetwork type ANN {
    ...
}
simulate MyNetwork;
'''
interpret_neuroscript(dsl_code)
```

---

## Training Methods

NeuroScript supports various training methods through DSL parameters:
- **Optimizers:** `Adam`, `SGD`, etc.
- **Learning Rate:** Specify using `lr`.
- **Schedulers:** For example, `StepLR` with `step_size` and `gamma`.
- **Additional Parameters:** For SGD, you can also provide `momentum`.

The generated Python code includes a training loop that uses the specified parameters.

---

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

---

## Conclusion

NeuroScript provides a high-level, human-readable way to define neural network architectures, training procedures, and data operations, which are automatically converted into executable Python code. This documentation outlines the DSL’s syntax and capabilities to help you get started.

For further details, please refer to the source code and examples provided in this repository.

