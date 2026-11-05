import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import main


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(Path(__file__).with_name('data.json').read_text(encoding='utf-8'))

    def analyze(self, data):
        with contextlib.redirect_stdout(io.StringIO()):
            return main.analyze(data, 10)

    def test_scores_and_generator(self):
        for n in (3, 5, 13, 25):
            cross, x = main.generate_patterns(n)
            self.assertEqual(main.mac(cross, cross), 2 * n - 1)
            self.assertEqual(main.mac(cross, x), 1)
            self.assertEqual(main.mac(x, x), 2 * n - 1)

    def test_labels(self):
        for value, expected in [('+', 'Cross'), ('cross', 'Cross'), ('Cross', 'Cross'), ('x', 'X'), ('X', 'X')]:
            self.assertEqual(main.normalize_label(value), expected)
        with self.assertRaises(ValueError):
            main.normalize_label('unknown')

    def test_epsilon(self):
        self.assertEqual(main.decide(.9, .8999999999999999), 'UNDECIDED')
        self.assertEqual(main.decide(0, 1e-9), 'X')
        self.assertEqual(main.decide(5, 1, ('A', 'B')), 'A')

    def test_sample(self):
        self.assertEqual(self.analyze(self.data), (6, []))

    def test_case_errors_do_not_stop_batch(self):
        data = copy.deepcopy(self.data)
        data['patterns']['size_5_1']['input'] = [[1]]
        data['patterns']['size_13_1']['expected'] = 'unknown'
        data['patterns']['bad_key'] = {}
        passed, failures = self.analyze(data)
        self.assertEqual(passed, 4)
        self.assertEqual(len(failures), 3)

    def test_missing_filter(self):
        del self.data['filters']['size_5']
        passed, failures = self.analyze(self.data)
        self.assertEqual((passed, len(failures)), (4, 2))

    def test_tie_is_failure(self):
        self.data['patterns']['size_5_1']['input'] = [[0] * 5 for _ in range(5)]
        passed, failures = self.analyze(self.data)
        self.assertEqual((passed, len(failures)), (5, 1))
        self.assertIn('동점', failures[0][1])

    def test_invalid_numeric_values(self):
        for value in (True, '1', float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                main.validate_matrix([[value]], 1)
        with self.assertRaises(ValueError):
            main.mac([[1e308]], [[1e308]])

    def test_input_retry(self):
        with patch('builtins.input', side_effect=['1 2', 'a b c', 'nan 1 2', '0 1 0', '1 1 1', '0 1 0']):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                result = main.read_matrix('test')
        self.assertEqual(result, main.generate_patterns(3)[0])
        self.assertEqual(output.getvalue().count('입력 형식 오류'), 3)


if __name__ == '__main__':
    unittest.main()
