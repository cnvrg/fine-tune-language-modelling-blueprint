import pandas as pd
import unittest
from finetune import DataTrainingArguments, get_dataset
import os
import pathlib as pl
import transformers
import shutil
import subprocess

from transformers import (
    AutoTokenizer,
)

print(os.getcwd())

class TestFT(unittest.TestCase):
    def assertFileExist(self, path):
        if not pl.Path(path):
            raise ValueError(
                "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                "or remove the --do_eval argument."
            )
    def assertDirExist(self, path):
        if (
            os.path.exists(path)
        ):
            raise ValueError(
                f"Output directory ({path}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

    # Test 1
    def test_data_paths(self):
        """Checks if input file existing"""
        file = pl.Path("./6_genre_clean_training_data_small.txt")
        self.assertTrue(file.resolve().is_file())


    # Test 2
    def test_data_paths(self):
        """Checks if input file existing"""
        file = pl.Path("./6_genre_eval_data_small.txt")
        self.assertTrue(file.resolve().is_file())

    # Test 3
    def test_eval_exit(self):
        """Checks if Eval file provided"""
        file = pl.Path("./6_genre_eval_data_small.txt")
        self.assertFileExist(file)

    # Test 4
    def test_outputdir_exist(self):
        """Checks if output_dir existing"""
        file = pl.Path("./story_generator_checkpoint_" + "bert-base-uncased")
        self.assertDirExist(file)

    # Test 5
    def test_return_type(self):
        """Checks if the function returns the type of transformers.data.datasets.language_modeling.LineByLineTextDataset"""
        data_args = DataTrainingArguments(
            train_data_file='unittests_training_data.txt',
            eval_data_file='unittests_eval_data.txt',
            line_by_line=True,
            block_size=256,

            overwrite_cache=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        returned_lists_trainy = get_dataset(data_args, tokenizer=tokenizer)
        returned_lists_evaly = get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
        self.assertIsInstance(returned_lists_trainy, transformers.data.datasets.language_modeling.LineByLineTextDataset)
        self.assertIsInstance(returned_lists_evaly, transformers.data.datasets.language_modeling.LineByLineTextDataset)


    # Test 6 Modified
    def test_return_metrics(self):
        """Checks if the metrics peplexity meet target value."""
        model_name = "bert-base-uncased"
        output_model_path = "story_generator_checkpoint_"
        bert_target = 220
        
        cmd = "python -u ./finetune.py --model_name bert-base-uncased "
        try:
            result = subprocess.check_output(cmd, shell=True)
            result = result.decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(f' returned error: {e.returncode}, output: {e.output.decode()}')
            return e.returncode
        perplexity = result.split("=")[1]
        message = "Perplexity is not less that target value."
        self.assertLess(float(perplexity), bert_target, message)
        if output_model_path+model_name:
            shutil.rmtree(output_model_path+model_name, ignore_errors=True)       
 
if __name__ == "__main__":
    unittest.main()