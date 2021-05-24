
import copy

PROXIES=dict()

def register(dst):
    '''Register a class to a dstination dictionary. '''
    def dec_register(cls):
        dst[cls.__name__] = cls
        return cls
    return dec_register

def make_object(typeD, argD):
    '''Make an object from type collection typeD. '''

    assert( isinstance(typeD, dict) ), f'typeD must be dict. typeD is {type(typeD)}'
    assert( isinstance(argD,  dict) ), f'argD must be dict. argD is {type(argD)}'
    
    # Make a deep copy of the input dict.
    d = copy.deepcopy(argD)

    try:
        # Get the type.
        typeName = typeD[ d['type'] ]

        # Remove the type string from the input dictionary.
        d.pop('type')
    except KeyError as exc:
        print('KeyError encountered when making an object. All the keys: ')
        print(PROXIES.keys())
        raise exc

    # Create the model.
    return typeName( **d )