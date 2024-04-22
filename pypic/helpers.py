import os
import sys
# Add parent package directory to path
# path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if path not in sys.path:
#     sys.path.append(path)


def print_var(var, name='', dp=2):
    print(f' * {name:10s} * | Min: {np.min(var):.2f} | Max: {np.max(var):.2f} | '
          +f'Med: {np.median(var):.2f} | Mean: {np.mean(var):.2f} | '
          +f'std: {np.std(var):.2f} | N: {np.size(var):,d}')


# MIN_VERSION_PY = (3, 8)
# if (sys.version_info < MIN_VERSION_PY):
#       sys.exit(
#         "ERROR: This script requires Python >= {};\n"
#         " you're running {}.".format(
#           '.'.join(map(str, MIN_VERSION_PY)),
#           '.'.join(map(str, sys.version_info))
#         )
#       )


# spam_spec = importlib.util.find_spec("pypic.geometry")
# spam_spec = importlib.util.find_spec("..geometry", package="pypic.bar")
# found = spam_spec is not None
# print(spam_spec.name == "pypic.geometry")
# print(f'{found=}')

# if 'pypic.geometry' in sys.modules:
#     print("Module is imported.")
# else:
#     print("Module is not imported.")

check_library_imports = False
if check_library_imports:
    import importlib.util
    # For illustration
    name = 'pypic.input_output'

    #code to check if the library exists
    if (spec := importlib.util.find_spec(name)) is not None:
        #displaying that the module is found
        # print(sys.modules)
        print(f"{name!r} already in sys.modules")
        #importing the present module
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        print('---')
        # print(sys.modules)
        #displaying that the module has been imported
        print(f"{name!r} has been imported")
    #else displaying that the module is absent
    else:
        print(f"can't find the {name!r} module")


# print(sys.implementation.name)
# print(sys.implementation.version)

# print(f'{path=}')

# for path in sys.path:
#     print(path)

# CURR_DIR = os.path.dirname(os.path.abspath(__file__))
# print(CURR_DIR)
# sys.path.append(CURR_DIR)
# for path in sys.path:
#     print(path)

# from ..pkg_a.mod_a import function_a
