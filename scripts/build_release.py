from release_common import freeze, launch_check, archive
folder = freeze('industrial_gui.py', 'MuScGUI', '--collect-all', 'models', '--collect-all', 'datasets',
                '--collect-all', 'utils', '--collect-all', 'timm', '--collect-all', 'transformers',
                '--collect-all', 'open_clip',
                '--collect-all', 'torch', '--collect-all', 'torchvision',
                '--add-data', 'config.yaml;.', '--add-data', 'configs;configs', '--add-data', 'LICENSE;.')
launch_check(folder / 'MuScGUI.exe', '--smoke-test')
archive(folder, 'musc-gui-windows.zip')
