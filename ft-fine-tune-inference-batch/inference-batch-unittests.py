import pandas as pd
import unittest
from inferencebatch import predict
import os
import pathlib as pl
# from pathlib import Path
# import transformers
from transformers import PreTrainedTokenizer, LineByLineTextDataset, TextDataset, AutoTokenizer, TextGenerationPipeline, BertLMHeadModel,GPT2LMHeadModel
import sys

print(os.getcwd())

class TestFT(unittest.TestCase):
    def setUp(self):
        """Overrides setUp from unittest to create a toy pandas dataframe for testing"""
        # Here anomalies and non-anomalies are represented by -1 and 1 respectively.
        self.train_data = [
            [
                "<BOS> <action> Prince (Vivek Oberoi) is"
            ],
            [
                "<BOS> <drama> Life is going along"
            ],
            [
                "<BOS> <drama> A young girl suddenly"
            ],
            [
                "<BOS> <drama> Varghese (Mohanlal) is a"
            ],
            [
                "<BOS> <drama> Liz is a Los"
            ],
         ]

        self.train_data_list = []
        for row in self.train_data:
            for col in row: 
                self.train_data_list.append(str(col))
        # print("===self.train_data_list===", self.train_data_list)

        # Expected values
        self.story_result = [100, 100, 100, 100, 100]


    def assertIsDir(self, path):
        if not pl.Path(path).resolve().is_dir():
            raise AssertionError("Folder does not exist: %s" % str(path))

    # Test 1
    def test_data_paths(self):
        """Checks if model existing"""
        path = pl.Path('./story_generator_checkpoint_gpt2')
        self.assertIsDir(path)

    # Test 2
    def test_return_length(self):
        """Checks if the function returns the correct length of tories generated, less or equal than required length"""
        scripts_dir = pl.Path(__file__).parent.resolve()
        sys.path.append(str(scripts_dir))
        length_story = 100
        model_path = './story_generator_checkpoint_' 
        model_name = 'gpt2'

        path_name = model_path + model_name
        checkpoint = os.path.join(scripts_dir, path_name)   

        checkpoint_str = str(checkpoint)
        if "gpt2" in checkpoint_str or "distilgpt2" in checkpoint_str:
            model = GPT2LMHeadModel.from_pretrained(checkpoint)
        else:
            model = BertLMHeadModel.from_pretrained(checkpoint, is_decoder=True)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        story_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
        story = predict(story_generator, self.train_data_list, length_story)
        story_length = [len(item[0]['generated_text'].split()) for item in story]
        print("===story_length===", story_length)

        self.assertLess(story_length, self.story_result)


if __name__ == "__main__":
    unittest.main()