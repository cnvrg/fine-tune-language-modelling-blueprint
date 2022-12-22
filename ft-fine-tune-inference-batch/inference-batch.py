import numpy as np
import pandas as pd
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import BertTokenizer, BertForNextSentencePrediction, BertLMHeadModel, DataCollatorWithPadding, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextGenerationPipeline, pipeline
import re
import os
import pathlib
import argparse
import yaml
import sys
import shutil
import glob
from pathlib import Path

def parse_parameters():
    parser = argparse.ArgumentParser(description="""finetune pre-trained Huggingface bert model""")
    parser.add_argument('-input_filename', '--input_filename', action='store', dest='input_filename', default='./6_genre_clean_inference_data.txt', required=False,
                        help="""string. csv train data file""")

    parser.add_argument('-model_name', '--model_name', action='store', dest='model_name', default='None', required=False, 
                        help="""string. name for model""")
    
    parser.add_argument('-model_path', '--model_path', action='store', dest='model_path', default='./story_generator_checkpoint_', required=False, 
                        help="""string. path for model""")

    parser.add_argument('-best_model_path', '--best_model_path', action='store', dest='best_model_path', default="/input/dev_ft_compare_language_4/story_generator_checkpoint_*", required=False, 
                        help="""string. path for best model after compare""")

    parser.add_argument('-result_path', '--result_path', action='store',
                        default='/cnvrg', required=False,
                        help="""string. path for saving the result""")

    parser.add_argument('-text', '--text_column', action='store', dest='text_column', default='text', required=False,
                        help="""string. name of text column""")

    parser.add_argument('-length_story', '--length_story', action='store', dest='length_story', default=100, required=False,
                        help="""int. size of story length for each topic""")
    return parser.parse_args()

def predict(story_generator, input, length_story):
    # input_prompt = "<BOS> <superhero> Spiderman is a movie"
    story = story_generator(input, max_length=length_story, do_sample=True,
                repetition_penalty=1.1, temperature=1.2, 
                top_p=0.95, top_k=50, num_return_sequences=1)
    return story

# Define main function
def main():
    # Run these cells for story generation
    """ 
    Below, my model checkpoint is commented out. You can replace your checkpoint 
    with that to test story generation if your checkpoint didn't train for long enough
    """
    scripts_dir = pathlib.Path(__file__).parent.resolve()
    sys.path.append(str(scripts_dir))
    cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", scripts_dir)

    # Read config file
    # os.chdir("fine-tune-inference/")
    with open("./inference-batch-config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    args = parse_parameters()
    result_path = args.result_path
    text_column = args.text_column
    input_filename = args.input_filename
    model_name = args.model_name
    length_story = int(args.length_story)
    model_path = args.model_path
    best_model_path = args.best_model_path

    #checkpoint = "./results"
    if model_name == 'None':
        shutil.move(best_model_path, cnvrg_workdir)
        for path in glob.glob(cnvrg_workdir+"/story_generator_checkpoint_*"):
            checkpoint = path
            print("checkpoint path", checkpoint)  
    else:
        path_name = model_path + model_name
        model_path = os.path.join(scripts_dir, path_name)   
        checkpoint = model_path

    # if os.path.exists(best_model_path):
    #     shutil.move(best_model_path, cnvrg_workdir)
    #     for path in glob.glob(cnvrg_workdir+"/story_generator_checkpoint_*"):
    #         checkpoint = path
    #         print("checkpoint path", checkpoint)       
    # else: 
    #     path_name = model_path + model_name
    #     model_path = os.path.join(scripts_dir, path_name)   
    #     checkpoint = model_path

    # checkpoint = "fine-tune-inference/story_generator_checkpoint_gpt2"
    # checkpoint = "fine-tune-inference/story_generator_checkpoint_distilgpt2"
    # checkpoint = "fine-tune-inference/story_generator_checkpoint_bert"
    # print("PATH:",os.getcwd())
    
    checkpoint_str = str(checkpoint)
    if "gpt2" in checkpoint_str or "distilgpt2" in checkpoint_str:
        model = GPT2LMHeadModel.from_pretrained(checkpoint)
    else:
        model = BertLMHeadModel.from_pretrained(checkpoint, is_decoder=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # story_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    # The format for input_prompt: "<BOS> <genre> Optional text..."
    # Supported genres: superhero, sci_fi, horror, thriller, action, drama


    # Story Inference Example
    story_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    # Read Batch Predict Data
    DATASET_COLUMNS = config["DATASET_COLUMNS"]
    DATASET_ENCODING = config["DATASET_ENCODING"]
    test_data = pd.read_csv(input_filename, sep="<EOS>", header=None, index_col=False, encoding=DATASET_ENCODING, names=DATASET_COLUMNS, engine='python')
    test_data = test_data.head(5)
    test_data['text_to_predict'] = test_data[text_column].str.split().str[0:6].str.join(' ')
    input = list(test_data['text_to_predict'].astype(str))

    # Save the prediction result
    story = predict(story_generator, input, length_story)
    test_data[config["DATASET_PREDICTION"]] = [item[0]['generated_text'] for item in story]
    test_data[["text_to_predict", config["DATASET_PREDICTION"]]].to_csv(result_path+config["RESULT_FILE"], header=True, index=False)
   

if __name__ == "__main__":
    main()
