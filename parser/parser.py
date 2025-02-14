# parser/parser.py
import ply.yacc as yacc
from .lexer import tokens
from .ast import ASTNode

# Starting rule: a program is a list of statements
def p_program(p):
    '''program : statements'''
    p[0] = ASTNode("program", statements=p[1])

def p_statements_multiple(p):
    '''statements : statements statement'''
    p[0] = p[1] + [p[2]]

def p_statements_single(p):
    '''statements : statement'''
    p[0] = [p[1]]

# Statements: network definition, simulate, visualize, or extra commands
def p_statement_network(p):
    '''statement : network_def'''
    p[0] = p[1]

def p_statement_simulate(p):
    '''statement : simulate_stmt'''
    p[0] = p[1]

def p_statement_visualize(p):
    '''statement : visualize_stmt'''
    p[0] = p[1]

def p_statement_extra(p):
    '''statement : data_def
                 | save_model_stmt
                 | load_model_stmt
                 | predict_stmt'''
    p[0] = p[1]

# Network definition: network <name> type <type> { network_body }
def p_network_def(p):
    '''network_def : NETWORK IDENTIFIER TYPE IDENTIFIER LBRACE network_body RBRACE'''
    p[0] = ASTNode("network_def", name=p[2], net_type=p[4], body=p[6])

# Network body: list of network items
def p_network_body(p):
    '''network_body : network_body network_item
                    | empty'''
    if len(p) == 2:
        p[0] = []
    else:
        p[0] = p[1]
        p[0].append(p[2])

def p_network_item(p):
    '''network_item : layer_def
                    | generic_layer_def
                    | connect_def
                    | training_def
                    | simulation_def
                    | neuron_type_def
                    | data_def
                    | save_model_stmt
                    | load_model_stmt
                    | predict_stmt'''
    p[0] = p[1]

# Layer definitions for ANN: input_layer, hidden_layer, output_layer
def p_layer_def(p):
    '''layer_def : INPUT_LAYER LBRACE property_list RBRACE
                 | HIDDEN_LAYER LBRACE property_list RBRACE
                 | OUTPUT_LAYER LBRACE property_list RBRACE'''
    p[0] = ASTNode("layer_def", layer_type=p[1], properties=p[3])

# Generic layer definition (e.g., for SNN)
def p_generic_layer_def(p):
    '''generic_layer_def : LAYER IDENTIFIER LBRACE property_list RBRACE'''
    p[0] = ASTNode("generic_layer_def", name=p[2], properties=p[4])

# Connection definition: connect <source> -> <target> { property_list }
def p_connect_def(p):
    '''connect_def : CONNECT IDENTIFIER ARROW IDENTIFIER LBRACE property_list RBRACE'''
    p[0] = ASTNode("connect_def", source=p[2], target=p[4], properties=p[7])

# Training block: training { property_list }
def p_training_def(p):
    '''training_def : TRAINING LBRACE property_list RBRACE'''
    p[0] = ASTNode("training_def", properties=p[3])

# Simulation block: simulation { property_list }
def p_simulation_def(p):
    '''simulation_def : SIMULATION LBRACE property_list RBRACE'''
    p[0] = ASTNode("simulation_def", properties=p[3])

# Neuron type definition: neuron_type <name> { property_list }
def p_neuron_type_def(p):
    '''neuron_type_def : NEURON_TYPE IDENTIFIER LBRACE property_list RBRACE'''
    p[0] = ASTNode("neuron_type_def", name=p[2], properties=p[4])

# Data block: data { property_list }
def p_data_def(p):
    '''data_def : DATA LBRACE property_list RBRACE'''
    p[0] = ASTNode("data_def", properties=p[3])

# Save model statement: save_model "path" ;
def p_save_model_stmt(p):
    '''save_model_stmt : SAVE_MODEL STRING SEMICOLON'''
    p[0] = ASTNode("save_model_stmt", path=p[2])

# Load model statement: load_model "path" ;
def p_load_model_stmt(p):
    '''load_model_stmt : LOAD_MODEL STRING SEMICOLON'''
    p[0] = ASTNode("load_model_stmt", path=p[2])

# Predict statement: predict "input_path" ;
def p_predict_stmt(p):
    '''predict_stmt : PREDICT STRING SEMICOLON'''
    p[0] = ASTNode("predict_stmt", input_path=p[2])

# --- New rules for mathematical functions (vectors, matrices) ---

# Allow value to be a vector literal
def p_value_vector(p):
    'value : vector_literal'
    p[0] = p[1]

def p_vector_literal(p):
    'vector_literal : LSQ expr_list RSQ'
    p[0] = ('vector', p[2])

def p_expr_list_multiple(p):
    'expr_list : expr_list COMMA math_term'
    p[0] = p[1] + [p[3]]

def p_expr_list_single(p):
    'expr_list : math_term'
    p[0] = [p[1]]

def p_math_term_number(p):
    'math_term : NUMBER'
    p[0] = p[1]

# Allow value to be a matrix literal
def p_value_matrix(p):
    'value : matrix_literal'
    p[0] = p[1]

def p_matrix_literal(p):
    'matrix_literal : LSQ matrix_row_list RSQ'
    p[0] = ('matrix', p[2])

def p_matrix_row_list_multiple(p):
    'matrix_row_list : matrix_row_list COMMA vector_literal'
    p[0] = p[1] + [p[3]]

def p_matrix_row_list_single(p):
    'matrix_row_list : vector_literal'
    p[0] = [p[1]]

# Value can be a number, identifier, string, vector, or matrix
def p_value_number(p):
    '''value : NUMBER'''
    p[0] = p[1]

def p_value_identifier(p):
    '''value : IDENTIFIER'''
    p[0] = p[1]

def p_value_string(p):
    '''value : STRING'''
    p[0] = p[1]

# --- End of math functions rules ---

# Simulate statement: simulate <network_name> ;
def p_simulate_stmt(p):
    '''simulate_stmt : SIMULATE IDENTIFIER SEMICOLON'''
    p[0] = ASTNode("simulate_stmt", target=p[2])

# Visualize statement: visualize <network_name> ;
def p_visualize_stmt(p):
    '''visualize_stmt : VISUALIZE IDENTIFIER SEMICOLON'''
    p[0] = ASTNode("visualize_stmt", target=p[2])

# Empty production rule
def p_empty(p):
    'empty :'
    p[0] = []

def p_error(p):
    if p:
        print("Syntax error at token", p.type, "with value", p.value)
    else:
        print("Syntax error at EOF")

# Build the parser
parser = yacc.yacc()

if __name__ == "__main__":
    data = '''
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
        training {
            optimizer: Adam;
            loss: cross_entropy;
            epochs: 20;
            batch_size: 64;
        }
        data {
            train: "train_data.csv";
        }
        save_model "mymodel.pth";
        predict "predict_input.csv";
    }
    simulate MyNetwork;
    visualize MyNetwork;
    '''
    result = parser.parse(data)
    print(result)
