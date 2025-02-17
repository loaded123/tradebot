# analyze_imports.py
import modulefinder
import os

# Path to your main script
main_script = 'C:/Users/Dennis/.vscode/tradebot/src/models/train_lstm_model.py'

finder = modulefinder.ModuleFinder()
finder.run_script(main_script)

#print('Loaded modules:')
#for name, mod in finder.modules.items():
#    print(f'{name}: {mod.__file__}')

print('\nModules that import backtester:')
for name, mod in finder.modules.items():
    if 'backtester' in mod.globalnames:
        print(f'{name} imports backtester')
