# parser/lexer.py
import ply.lex as lex

# Dictionary of reserved words
reserved = {
    'network':      'NETWORK',
    'type':         'TYPE',
    'simulate':     'SIMULATE',
    'visualize':    'VISUALIZE',
    'input_layer':  'INPUT_LAYER',
    'hidden_layer': 'HIDDEN_LAYER',
    'output_layer': 'OUTPUT_LAYER',
    'neurons':      'NEURONS',
    'activation':   'ACTIVATION',
    'connect':      'CONNECT',
    'training':     'TRAINING',
    'optimizer':    'OPTIMIZER',
    'loss':         'LOSS',
    'epochs':       'EPOCHS',
    'batch_size':   'BATCH_SIZE',
    'neuron_type':  'NEURON_TYPE',
    'layer':        'LAYER',
    'simulation':   'SIMULATION',
    'duration':     'DURATION',
    'time_step':    'TIME_STEP',
    # New reserved words:
    'data':         'DATA',
    'save_model':   'SAVE_MODEL',
    'load_model':   'LOAD_MODEL',
    'predict':      'PREDICT',
}

# List of token names
tokens = [
    'IDENTIFIER',
    'NUMBER',
    'STRING',
    'COLON',
    'SEMICOLON',
    'COMMA',
    'LBRACE',
    'RBRACE',
    'LPAREN',
    'RPAREN',
    'ARROW',
] + list(reserved.values())

# Regular expressions for tokens
t_COLON      = r':'
t_SEMICOLON  = r';'
t_COMMA      = r','
t_LBRACE     = r'\{'
t_RBRACE     = r'\}'
t_LPAREN     = r'\('
t_RPAREN     = r'\)'
t_ARROW      = r'->'

# Token for string literals (double quotes)
def t_STRING(t):
    r'\"([^\\\n]|(\\.))*?\"'
    t.value = t.value[1:-1]  # remove quotes
    return t

# Ignored characters (spaces and tabs)
t_ignore = ' \t'

# Single-line comments (// comment)
def t_COMMENT(t):
    r'//.*'
    pass

# Token for numbers (integers and floats)
def t_NUMBER(t):
    r'\d+(\.\d+)?'
    if '.' in t.value:
        t.value = float(t.value)
    else:
        t.value = int(t.value)
    return t

# Token for identifiers with reserved word checking
def t_IDENTIFIER(t):
    r'[A-Za-z_][A-Za-z0-9_]*'
    t.type = reserved.get(t.value, 'IDENTIFIER')
    return t

# Count newlines
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# Error handling for illegal characters
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()

if __name__ == "__main__":
    data = 'network MyNetwork type ANN { input_layer { neurons: 784; } }'
    lexer.input(data)
    for token in lexer:
        print(token)
