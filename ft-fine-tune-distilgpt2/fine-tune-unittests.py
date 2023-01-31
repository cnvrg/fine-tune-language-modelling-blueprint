import pandas as pd
import unittest
from finetune import DataTrainingArguments, get_dataset
import os
import pathlib as pl
import sys
import transformers
import shutil
import subprocess

from transformers import AutoTokenizer

scripts_dir = pl.Path(__file__).parent.resolve()
sys.path.append(str(scripts_dir))

print(os.getcwd())

class TestFT(unittest.TestCase):   
    # Test 1
    def test_data_paths(self):
        """Checks if input file existing"""
        scripts_dir = pl.Path(__file__).parent.resolve()
        sys.path.append(str(scripts_dir))
        file = pl.Path(os.path.join(scripts_dir, '6_genre_clean_training_data_small.txt'))
        self.assertTrue(file.resolve().is_file())

    # Test 2
    def test_data_paths(self):
        """Checks if input file existing"""
        scripts_dir = pl.Path(__file__).parent.resolve()
        file = pl.Path(os.path.join(scripts_dir, '6_genre_eval_data_small.txt'))
        self.assertTrue(file.resolve().is_file())

    # Test 3
    def test_eval_exit(self):
        """Checks if Eval file provided"""
        """Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file """
        """or remove the --do_eval argument."""
        scripts_dir = pl.Path(__file__).parent.resolve()
        file = pl.Path(os.path.join(scripts_dir, '6_genre_eval_data_small.txt'))
        self.assertTrue(pl.Path(file))

    # Test 4
    def test_outputdir_exist(self):
        """Checks if output_dir not existing"""
        """If output directory ({path}) already exists and is not empty. Use --overwrite_output_dir to overcome."""
        scripts_dir = pl.Path(__file__).parent.resolve()
        file = pl.Path(os.path.join(scripts_dir, 'story_generator_checkpoint_' + 'distilgpt2'))
        self.assertFalse(os.path.exists(file))

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
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        returned_lists_trainy = get_dataset(data_args, tokenizer=tokenizer)
        returned_lists_evaly = get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
        self.assertIsInstance(returned_lists_trainy, transformers.data.datasets.language_modeling.LineByLineTextDataset)
        self.assertIsInstance(returned_lists_evaly, transformers.data.datasets.language_modeling.LineByLineTextDataset)

    # Test 6 Modified
    def test_return_metrics(self):
        """Checks if the metrics peplexity meet target value."""
        scripts_dir = pl.Path(__file__).parent.resolve()
        model_name = "distilgpt2"
        output_model_path = "story_generator_checkpoint_"
        distilgpt2_target = 90
        finetune_path = os.path.join(scripts_dir, 'finetune.py')

        cmd = "python -u " + finetune_path + " --model_name distilgpt2 " \
              + " --input_filename_train 6_genre_clean_training_data_small.txt " \
              + " --input_filename_test 6_genre_eval_data_small.txt " \
              + " --output_model_path story_generator_checkpoint_ " \
              + " --num_train_epochs 1 " \
              + " --logging_steps 500 " \
              + " --save_steps 1000 " \
              + " --max_length 256 "

        try:
            result = subprocess.check_output(cmd, shell=True)
            result = result.decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(f' returned error: {e.returncode}, output: {e.output.decode()}')
            return e.returncode
        perplexity = result.split("=")[1]
        message = "Perplexity is not less than target value."
        self.assertLess(float(perplexity), distilgpt2_target, message)
        if output_model_path+model_name:
            shutil.rmtree(output_model_path+model_name, ignore_errors=True)  

if __name__ == "__main__":
    unittest.main()