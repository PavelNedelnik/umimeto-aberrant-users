import os
import subprocess
import json
from io import StringIO
from pathlib import Path
from shutil import rmtree
from pylint.lint import Run
from pylint.reporters.json_reporter import JSONReporter
from tqdm import tqdm


def edulint_analyze(file_path):
    result = subprocess.run(['py', '-m', 'edulint', str(file_path)], text=True, capture_output=True)
    return [msg[msg.rfind(':') + 2:msg.find('[')].replace(' ', '_') for msg in result.stdout.split('\n') if len(msg) > 0]


def pylint_analyze(file_path):
    pylint_output = StringIO()
    reporter = JSONReporter(pylint_output)
    result = Run([str(file_path)], reporter=reporter, exit=False)
    return [error.symbol for error in result.linter.reporter.messages]


def flake8_analyze(file_path):
    raise RuntimeError('Not implemented!')
    result = subprocess.run(['flake8', '-'], input=code_string, text=True, capture_output=True)
    return result.stdout.split('\n')


def call_analyze(file_path, mode):
    if mode == 'edulint':
        return edulint_analyze(file_path)
    elif mode == 'pylint':
        return pylint_analyze(file_path)
    elif mode == 'flake8':
        return flake8_analyze(file_path)
    
    raise RuntimeError('Mode not recognized!')


def analyze_string(code_string, mode='edulint', file_path='temp_code.py'):
    with open(file_path, 'w') as f:
        f.write(code_string)

    result = call_analyze(file_path, mode)

    os.remove(file_path)
    return result


def analyze_strings(code_strings, mode='edulint', result_path='results.json'):
    if mode != 'edulint':
        raise RuntimeError('Not implemented!')
    
    print('Creating code files...')
    dir_path = Path('temp_folder/')
    dir_path.mkdir(parents=True, exist_ok=True)
    for i, code_string in tqdm(enumerate(code_strings)):
        file_path = dir_path / 'temp_code_{}.py'.format(i)
        with open(file_path, mode='w') as f:
            f.write(code_string)
    print('Done!')
    
    print('Processing files...')
    temp_results_path = dir_path / 'results.txt'
    max_i = i
    with open(temp_results_path, mode='w') as temp_results:
        for i in tqdm(range(max_i + 1)):
            file_path = dir_path / 'temp_code_{}.py'.format(i)
            temp_results.write(json.dumps(call_analyze(file_path, mode)))
            temp_results.write(';')
    print('Done!')

    print('Cleaning up...')
    with open(temp_results_path, mode='r') as temp_results:
        result = list(map(lambda x: [msg[:msg.find('_')] for msg in json.loads(x + ']')], temp_results.read().split('];')[:-1]))
        json.dump(result, open(result_path, 'w'))
    rmtree(dir_path)
    print('Done!')

    print('All finished!')

    return result