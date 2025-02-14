# transpiler/snn_generator.py

def generate_snn_code(network_ast):
    """
    Generates Python code for a Spiking Neural Network (SNN) using Brian2 based on the AST from NeuroScript DSL.
    
    Expects network_ast of type "network_def" with net_type "SNN" containing:
      - neuron_type_def: definitions of neuron types (with parameters threshold, reset_potential, refractory_period)
      - generic_layer_def: definitions of layers (parameters: neurons, type)
      - connect_def: connection definitions between layers (parameters: delay, weight)
      - simulation_def: simulation parameters (duration, time_step)
      Additional commands (data, save_model, load_model, predict) are processed similarly.
    
    Returns:
      A string containing the generated Python code using Brian2.
    """
    body = network_ast.children.get("body", [])
    
    neuron_types = {}
    layers = []
    connections = []
    simulation = None
    extra_nodes = []
    
    for node in body:
        if node.node_type == "neuron_type_def":
            nt_name = node.children.get("name")
            props = { key: value for key, value in node.children.get("properties", []) }
            neuron_types[nt_name] = props
        elif node.node_type == "generic_layer_def":
            layer_name = node.children.get("name")
            props = { key: value for key, value in node.children.get("properties", []) }
            layers.append({
                "name": layer_name,
                "neurons": props.get("neurons"),
                "neuron_type": props.get("type")
            })
        elif node.node_type == "connect_def":
            props = { key: value for key, value in node.children.get("properties", []) }
            connections.append({
                "source": node.children.get("source"),
                "target": node.children.get("target"),
                "delay": props.get("delay"),
                "weight": props.get("weight")
            })
        elif node.node_type == "simulation_def":
            simulation = { key: value for key, value in node.children.get("properties", []) }
        elif node.node_type in {"data_def", "save_model_stmt", "load_model_stmt", "predict_stmt"}:
            extra_nodes.append(node)

    code_lines = []
    code_lines.append("from brian2 import *")
    code_lines.append("")
    
    # Define the neuron model template
    code_lines.append("model_template = '''")
    code_lines.append("dv/dt = (v_rest - v) / tau : volt")
    code_lines.append("'''")
    code_lines.append("")
    
    code_lines.append("neuron_types = {")
    for nt_name, props in neuron_types.items():
        threshold   = props.get("threshold", "None")
        reset       = props.get("reset_potential", "None")
        refractory  = props.get("refractory_period", "None")
        code_lines.append(f"    '{nt_name}': {{")
        code_lines.append(f"         'threshold': {threshold},")
        code_lines.append(f"         'reset': {reset},")
        code_lines.append(f"         'refractory': {refractory}")
        code_lines.append("    },")
    code_lines.append("}")
    code_lines.append("")
    
    # Generate definitions for layers as NeuronGroups
    for layer in layers:
        layer_name   = layer["name"]
        neurons      = layer["neurons"]
        neuron_type  = layer["neuron_type"]
        code_lines.append(f"{layer_name} = NeuronGroup({neurons}, model=model_template,")
        code_lines.append(f"    threshold=neuron_types['{neuron_type}']['threshold'],")
        code_lines.append(f"    reset=neuron_types['{neuron_type}']['reset'],")
        code_lines.append(f"    refractory=neuron_types['{neuron_type}']['refractory'],")
        code_lines.append("    method='exact')")
        code_lines.append("")
    
    # Generate connection definitions as Synapses
    for conn in connections:
        src    = conn["source"]
        tgt    = conn["target"]
        delay  = conn["delay"]
        weight = conn["weight"]
        synapse_name = f"S_{src}_{tgt}"
        code_lines.append(f"{synapse_name} = Synapses({src}, {tgt}, on_pre='v_post += {weight}', delay={delay})")
        code_lines.append(f"{synapse_name}.connect()")
        code_lines.append("")
    
    if simulation:
        duration = simulation.get("duration", "100*ms")
        time_step = simulation.get("time_step", None)
        if time_step:
            code_lines.append(f"defaultclock.dt = {time_step}")
        code_lines.append(f"run({duration})")
    else:
        code_lines.append("run(100*ms)  # Using default simulation time")
    
    # Process extra nodes (data, save_model, load_model, predict) similarly
    for node in extra_nodes:
        if node.node_type == "data_def":
            data_props = { k: v for k,v in node.children.get("properties", []) }
            if "train" in data_props:
                train_path = data_props["train"]
                code_lines.append("")
                code_lines.append("# Data loading not typically used in Brian2 simulation, but provided here as an example")
                code_lines.append("import pandas as pd")
                code_lines.append(f"train_df = pd.read_csv('{train_path}')")
                code_lines.append("print('Loaded training data for SNN (example):', train_df.head())")
        elif node.node_type == "save_model_stmt":
            save_path = node.children.get("path")
            code_lines.append(f'\n# Save SNN model parameters (example)')
            code_lines.append(f'with open("{save_path}", "w") as f:')
            code_lines.append("    f.write('SNN model parameters would be saved here')")
        elif node.node_type == "load_model_stmt":
            load_path = node.children.get("path")
            code_lines.append(f'\n# Load SNN model parameters (example)')
            code_lines.append(f'with open("{load_path}", "r") as f:')
            code_lines.append("    params = f.read()")
            code_lines.append("    print('Loaded SNN parameters:', params)")
        elif node.node_type == "predict_stmt":
            input_path = node.children.get("input_path")
            code_lines.append("")
            code_lines.append("import pandas as pd")
            code_lines.append(f"predict_df = pd.read_csv('{input_path}')")
            code_lines.append("# Prediction for SNN simulation is not standard; this is just an example")
            code_lines.append("print('Prediction input:', predict_df.head())")
    
    return "\n".join(code_lines)


if __name__ == "__main__":
    from parser.ast import ASTNode

    dummy_ast = ASTNode(
        "network_def",
        name="BioNet",
        net_type="SNN",
        body=[
            ASTNode("neuron_type_def", name="Excitatory", properties=[
                ("threshold", "-55*mV"),
                ("reset_potential", "-70*mV"),
                ("refractory_period", "5*ms")
            ]),
            ASTNode("neuron_type_def", name="Inhibitory", properties=[
                ("threshold", "-50*mV"),
                ("reset_potential", "-65*mV"),
                ("refractory_period", "2*ms")
            ]),
            ASTNode("generic_layer_def", name="ExcitatoryLayer", properties=[
                ("neurons", 100),
                ("type", "Excitatory")
            ]),
            ASTNode("generic_layer_def", name="InhibitoryLayer", properties=[
                ("neurons", 30),
                ("type", "Inhibitory")
            ]),
            ASTNode("connect_def", source="ExcitatoryLayer", target="InhibitoryLayer", properties=[
                ("delay", "1*ms"),
                ("weight", "0.5")
            ]),
            ASTNode("connect_def", source="InhibitoryLayer", target="ExcitatoryLayer", properties=[
                ("delay", "0.5*ms"),
                ("weight", "-0.7")
            ]),
            ASTNode("simulation_def", properties=[
                ("duration", "500*ms"),
                ("time_step", "0.1*ms")
            ]),
            ASTNode("data_def", properties=[("train", "train_data.csv")]),
            ASTNode("save_model_stmt", path="snnmodel.txt"),
            ASTNode("predict_stmt", input_path="predict_input.csv")
        ]
    )

    generated_code = generate_snn_code(dummy_ast)
    print("Generated Python code for SNN:")
    print("-------------------------------------")
    print(generated_code)
