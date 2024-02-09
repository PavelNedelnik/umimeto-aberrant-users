import unittest
from pathlib import Path
from src.load_scripts import load_ipython_item, load_ipython_log

class CodeProcessingTest(unittest.TestCase):

    def test_decode_string(self):
        pass


class IpythonLoadingTest(unittest.TestCase):

    def test_load_sanity_check(self):
        path = Path('data/')
        self.assertFalse(load_ipython_item(path).empty, 'Nothing returned')
        self.assertFalse(load_ipython_log(path).empty, 'Nothing returned')
        # TODO self.assertTrue(load_ipython_log(path, linter_messages_path), 'Nothing returned')


    def test_load_dummy_dataset(self):
        pass