import unittest
from pathlib import Path
from src.load_scripts import load_ipython_item, load_ipython_log

class CodeProcessingTest(unittest.TestCase):

    def test_decode_string(self):
        pass


class IpythonLoadingTest(unittest.TestCase):


    def test_load_dummy_data(self):
        path = Path('tests/dummy_data/')
        log = load_ipython_log(path)
        item = load_ipython_item(path)

        # check empty
        self.assertFalse(log.empty, 'Empty log')
        self.assertFalse(item.empty, 'Empty item')

        # check columns
        expected_log_columns = [
            'id', 'user', 'item', 'answer', 'correct', 'moves', 'responseTime', 'time'
        ]
        self.assertEqual(log.columns.to_list(), expected_log_columns, 'Unexpected log columns')
        expected_item_columns = ['name', 'instructions', 'solution']
        self.assertEqual(item.columns.to_list(), expected_item_columns, 'Unexpected item columns')

        # check rows
        self.assertEqual(len(log), 6, 'Wrong log length')
        self.assertEqual(len(item), 10, 'Wrong item length')


    def test_load_real_data(self):
        path = Path('data/')

        item = load_ipython_item(path)
        log = load_ipython_log(path)

        # check empty
        self.assertFalse(item.empty, 'Empty log')
        self.assertFalse(log.empty, 'Empty item')

        # check null-values
        self.assertEqual(log.isnull().sum().sum(), 0, 'Found null values in log')
        self.assertEqual(item.isnull().sum().sum(), 0, 'Found null values in item')


    def test_linter_messages(self):
        # self.assertTrue(load_ipython_log(path, linter_messages_path), 'Nothing returned')
        # TODO features
        pass
