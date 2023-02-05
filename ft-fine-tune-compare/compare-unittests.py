import pandas as pd
import unittest
from compare import evaluation
import os
import pathlib as pl


print(os.getcwd())

class TestFT(unittest.TestCase):
    def setUp(self):
        """Overrides setUp from unittest to create a toy pandas dataframe for testing"""
        # Here anomalies and non-anomalies are represented by -1 and 1 respectively.
        self.model_name_1 = 'bert-base-uncased'
        self.model_name_2 = 'distilgpt2'
        self.model_name_3 = 'gpt2'

        self.eval_results_lm_1 = [
            [
                "perplexity = 12.804767494705949"
            ],
         ]

        self.eval_results_lm_2 = [
            [
                "perplexity = 47.756939755283454"
            ],
         ]

        self.eval_results_lm_3 = [
            [
                "perplexity = 37.34255788762836"
            ],
         ]

        with open("eval_results_lm_1.txt", "w") as output:
            for row in self.eval_results_lm_1:
                for col in row: 
                    output.write(str(col) + '\n')

        with open("eval_results_lm_2.txt", "w") as output:
            for row in self.eval_results_lm_2:
                for col in row: 
                    output.write(str(col) + '\n')

        with open("eval_results_lm_3.txt", "w") as output:
            for row in self.eval_results_lm_3:
                for col in row: 
                    output.write(str(col) + '\n')

        self.filelist = [self.model_name_1, self.model_name_2, self.model_name_3]

        self.filesList = ['eval_results_lm_1.txt', 
                          'eval_results_lm_2.txt', 
                          'eval_results_lm_3.txt']

        # Expected values
        self.best_model = 'bert-base-uncased'
        self.best_model_index = 0

    def assertIsFile(self, path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))
    
    # Test 1
    def test_data_paths(self):
        """Checks if input file existing"""
        file = pl.Path("eval_results_lm_1.txt")
        self.assertIsFile(file)

    # Test 2
    def test_data_paths(self):
        """Checks if input file existing"""
        file = pl.Path("eval_results_lm_2.txt")
        self.assertIsFile(file)

    # Test 3
    def test_data_paths(self):
        """Checks if input file existing"""
        file = pl.Path("eval_results_lm_3.txt")
        self.assertIsFile(file)

    # Test 4
    def test_result_values(self):
        """Checks if function returns the model with minimum perplexity"""
        returned_best_model, returned_best_model_index = evaluation(self.filelist, self.filesList)
        # print("===returned_best_model===", returned_best_model)
        self.assertEqual(returned_best_model, self.best_model)
        self.assertEqual(returned_best_model_index, self.best_model_index)
        if "eval_results_lm_1.txt" and "eval_results_lm_2.txt" and "eval_results_lm_3.txt":
            os.remove("eval_results_lm_1.txt")
            os.remove("eval_results_lm_2.txt")
            os.remove("eval_results_lm_3.txt")
        
if __name__ == "__main__":
    unittest.main()