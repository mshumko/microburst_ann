import sys
import pathlib

# Run the configuration script when the user runs 
# python3 -m microburst_ann [init, config, or configure]

here = pathlib.Path(__file__).parent.resolve()


if (len(sys.argv) > 1) and (sys.argv[1] in ['init', 'initialize', 'config', 'configure']):
    print('Running the configuration script.')
    # SAMPEX Data dir
    s = (f'What is the SAMPEX data directory? (It must contain the '
         f'attitude and hilt sub-directories) ')
    HILT_DIR = input(s)
    
    # Check that the SAMPEX directory exists
    if not pathlib.Path(HILT_DIR).exists():
        raise OSError(f'The HILT diretory "{HILT_DIR}" does not exist. Exiting.')
    
    with open(pathlib.Path(here, 'config.py'), 'w') as f:
        f.write('import pathlib\n\n')
        f.write(f'SAMPEX_DIR = pathlib.Path("{HILT_DIR}")\n')
        f.write(f'PROJECT_DIR = pathlib.Path("{here}")')

else:
    print('This is a configuration script to set up config.py file. The config '
        'file will contain the SAMPEX/HILT data directory, and the base project '
        'directory (here). To get the prompt after this package is installed, run '
        'python3 -m microburst_ann init')
