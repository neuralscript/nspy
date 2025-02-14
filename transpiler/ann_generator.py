# transpiler/ann_generator.py

def generate_ann_code(network_ast):
    """
    Generates Python code for an Artificial Neural Network (ANN) based on the AST produced from NeuroScript DSL.
    
    Args:
      network_ast: An ASTNode of type "network_def". Expected to contain:
         - name: the network name
         - net_type: the network type (e.g., "ANN")
         - body: a list of elements (layer definitions, connections, training parameters, etc.)
         
    Returns:
      A string containing the generated Python code.
    """
    # Extract the network name and body from the AST
    network_name = network_ast.children.get("name")
    body = network_ast.children.get("body", [])
    
    # Collect layer definitions, training parameters, and extra commands
    layer_nodes = []
    training_node = None
    extra_nodes = []  # For data, save_model, load_model, predict commands
    for node in body:
        if node.node_type == "layer_def":
            layer_nodes.append(node)
        elif node.node_type == "training_def":
            training_node = node
        elif node.node_type in {"data_def", "save_model_stmt", "load_model_stmt", "predict_stmt"}:
            extra_nodes.append(node)

    # Extract parameters for each layer
    layers_info = []
    for layer in layer_nodes:
        props = { key: value for key, value in layer.children.get("properties", []) }
        neurons = props.get("neurons")
        activation = props.get("activation", None)
        layers_info.append({
            "layer_type": layer.children.get("layer_type"),  # input_layer, hidden_layer, output_layer
            "neurons": neurons,
            "activation": activation
        })

    code_lines = []
    code_lines.append("import torch")
    code_lines.append("import torch.nn as nn")
    code_lines.append("import torch.optim as optim")
    code_lines.append("import torch.nn.functional as F")
    code_lines.append("")
    
    # Define the network class
    code_lines.append(f"class {network_name}(nn.Module):")
    code_lines.append("    def __init__(self):")
    code_lines.append(f"        super({network_name}, self).__init__()")
    
    # Generate layer definitions assuming sequential connection between layers
    num_layers = len(layers_info)
    for i in range(num_layers - 1):
        in_features = layers_info[i]["neurons"]
        out_features = layers_info[i+1]["neurons"]
        code_lines.append(f"        self.fc{i+1} = nn.Linear({in_features}, {out_features})")
    
    code_lines.append("")
    code_lines.append("    def forward(self, x):")
    code_lines.append("        # Forward propagation")
    if num_layers > 1:
        code_lines.append("        x = self.fc1(x)")
        act = layers_info[1].get("activation")
        if act:
            if act.lower() == "relu":
                code_lines.append("        x = torch.relu(x)")
            elif act.lower() == "softmax":
                code_lines.append("        x = torch.softmax(x, dim=1)")
            else:
                code_lines.append(f"        # Activation function '{act}' not implemented")
        for i in range(2, num_layers):
            code_lines.append(f"        x = self.fc{i}(x)")
            act = layers_info[i].get("activation")
            if act:
                if act.lower() == "relu":
                    code_lines.append("        x = torch.relu(x)")
                elif act.lower() == "softmax":
                    code_lines.append("        x = torch.softmax(x, dim=1)")
                else:
                    code_lines.append(f"        # Activation function '{act}' not implemented")
    else:
        code_lines.append("        # No transformations defined, returning input")
    code_lines.append("        return x")
    code_lines.append("")
    
    # Create an instance of the model, optimizer, and loss function
    code_lines.append(f"model = {network_name}()")
    if training_node:
        train_props = { key: value for key, value in training_node.children.get("properties", []) }
        optimizer_name = train_props.get("optimizer", "Adam")
        lr = train_props.get("lr", 0.001)
        # Create optimizer with specified learning rate and any additional parameters (e.g., momentum)
        if optimizer_name.lower() == "sgd":
            momentum = train_props.get("momentum", 0.9)
            code_lines.append(f"optimizer = optim.SGD(model.parameters(), lr={lr}, momentum={momentum})")
        else:
            code_lines.append(f"optimizer = optim.{optimizer_name}(model.parameters(), lr={lr})")
        if "scheduler" in train_props:
            scheduler_type = train_props.get("scheduler")
            if scheduler_type.lower() == "steplr":
                step_size = train_props.get("step_size", 10)
                gamma = train_props.get("gamma", 0.1)
                code_lines.append(f"scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size={step_size}, gamma={gamma})")
        loss_value = train_props.get("loss", "cross_entropy")
        if isinstance(loss_value, str) and loss_value.lower() == "cross_entropy":
            code_lines.append("criterion = nn.CrossEntropyLoss()")
        else:
            code_lines.append("criterion = None  # Loss function not implemented")
    else:
        code_lines.append("# Training parameters not provided")
    
    # Process extra nodes: data, save_model, load_model, predict
    for node in extra_nodes:
        if node.node_type == "data_def":
            data_props = { k: v for k, v in node.children.get("properties", []) }
            if "train" in data_props:
                train_path = data_props["train"]
                code_lines.append("")
                code_lines.append("import pandas as pd")
                code_lines.append("from torch.utils.data import DataLoader, TensorDataset")
                code_lines.append(f"train_df = pd.read_csv('{train_path}')")
                code_lines.append("# Assume the last column is the target and the rest are features")
                code_lines.append("inputs = torch.tensor(train_df.iloc[:, :-1].values, dtype=torch.float32)")
                code_lines.append("targets = torch.tensor(train_df.iloc[:, -1].values, dtype=torch.long)")
                code_lines.append("dataset = TensorDataset(inputs, targets)")
                batch_size = train_props.get("batch_size", 32)
                code_lines.append(f"train_loader = DataLoader(dataset, batch_size={batch_size}, shuffle=True)")
                epochs = train_props.get("epochs", 1)
                code_lines.append("")
                code_lines.append("for epoch in range({}):".format(epochs))
                code_lines.append("    for batch_inputs, batch_targets in train_loader:")
                code_lines.append("        optimizer.zero_grad()")
                code_lines.append("        outputs = model(batch_inputs)")
                code_lines.append("        loss_value = criterion(outputs, batch_targets)")
                code_lines.append("        loss_value.backward()")
                code_lines.append("        optimizer.step()")
                if "scheduler" in train_props:
                    code_lines.append("    scheduler.step()")
                code_lines.append("    print(f'Epoch {epoch+1}/{}, Loss: {loss_value.item()}')".format(epochs))
        elif node.node_type == "save_model_stmt":
            save_path = node.children.get("path")
            code_lines.append(f'\ntorch.save(model.state_dict(), "{save_path}")')
        elif node.node_type == "load_model_stmt":
            load_path = node.children.get("path")
            code_lines.append(f'\nmodel.load_state_dict(torch.load("{load_path}"))')
        elif node.node_type == "predict_stmt":
            input_path = node.children.get("input_path")
            code_lines.append("")
            code_lines.append("import pandas as pd")
            code_lines.append(f"predict_df = pd.read_csv('{input_path}')")
            code_lines.append("predict_inputs = torch.tensor(predict_df.values, dtype=torch.float32)")
            code_lines.append("predictions = model(predict_inputs)")
            code_lines.append("print('Predictions:', predictions)")
    
    return "\n".join(code_lines)


if __name__ == "__main__":
    from parser.ast import ASTNode
    
    # Dummy AST for demonstration purposes
    dummy_ast = ASTNode(
        "network_def",
        name="MyNetwork",
        net_type="ANN",
        body=[
            ASTNode("layer_def", layer_type="input_layer", properties=[("neurons", 784)]),
            ASTNode("layer_def", layer_type="hidden_layer", properties=[("neurons", 128), ("activation", "relu")]),
            ASTNode("layer_def", layer_type="output_layer", properties=[("neurons", 10), ("activation", "softmax")]),
            ASTNode("training_def", properties=[
                ("optimizer", "Adam"), 
                ("lr", 0.001), 
                ("loss", "cross_entropy"),
                ("epochs", 5), 
                ("batch_size", 64)
            ]),
            ASTNode("data_def", properties=[("train", "train_data.csv")]),
            ASTNode("save_model_stmt", path="mymodel.pth"),
            ASTNode("predict_stmt", input_path="predict_input.csv")
        ]
    )
    
    generated_code = generate_ann_code(dummy_ast)
    print("Generated Python code for ANN:")
    print("-----------------------------------")
    print(generated_code)
