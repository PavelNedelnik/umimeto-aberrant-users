import os
import subprocess
from io import StringIO
from pylint.lint import Run
from pylint.reporters.json_reporter import JSONReporter


def pylint_analysis(code_string):
    with open("temp_code.py", "w") as file:
        file.write(code_string)
    pylint_output = StringIO()  # Custom open stream
    reporter = JSONReporter(pylint_output)
    results = Run(["temp_code.py"], reporter=reporter, exit=False)
    os.remove("temp_code.py")
    return [error.symbol for error in results.linter.reporter.messages]


def flake8_analysis(code_string):
    result = subprocess.run(['flake8', '-'], input=code_string, text=True, capture_output=True)
    return result.stdout.split('\n')