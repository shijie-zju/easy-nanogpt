
'''
from train.py:
>>> exec(open('configurator.py').read())
'''


import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    # python script.py config/traingpt.py --key1=value1 --key2=value2 则sys.argv=['script.py', 'param1', 'param2', 'param3']
    if '=' not in arg:
    #first:open and exc the config/traingpt.py
    #the first as train.py and later is config/gpt1.py instead of --batch_size...
        print(f'arg:{arg}')
        assert not arg.startswith('--')
        config_file = arg
        print(f'overriding config with {config_file}')
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
    # later: --key1=value1
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:] # no --
        if key in globals():
            try:
                # attempt to eval it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val

            assert type(attempt) == type(globals()[key])

            print(f'Overriding: {key} = {attempt}')
            globals()[key] = attempt
        else:
            raise ValueError(f'Unknown config key: {key}')




