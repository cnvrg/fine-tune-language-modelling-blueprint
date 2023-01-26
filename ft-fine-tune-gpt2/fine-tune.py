import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForNextSentencePrediction, BertLMHeadModel
from transformers import AutoModelForCausalLM, TextGenerationPipeline
from transformers import EarlyStoppingCallback
import re
#import modeling
import os
import argparse
import logging

from datasets import load_dataset
from torch.nn import functional as F
import math
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    GPT2LMHeadModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)


# os.chdir("fine-tune/")
def parse_parameters():
    parser = argparse.ArgumentParser(description="""chatbot language modelling - finetune pre-trained Huggingface bert/gpt2/distilgpt2 models""")
    parser.add_argument('--input_filename_train', dest='input_filename_train', required=False,
                        help="""string. csv train data file""", default='6_genre_clean_training_data_small.txt')

    parser.add_argument('--input_filename_test', dest='input_filename_test', required=False,
                        help="""string. csv test data file""", default='6_genre_eval_data_small.txt')

    parser.add_argument('--model_name', default='gpt2', required=False,
                        help="""model to choose""")

    parser.add_argument('--output_model_path', default="story_generator_checkpoint_", required=False,
                        help="""string. path for saving the model""")

    parser.add_argument('--num_train_epochs', dest='num_train_epochs', default=1, required=False,
                        help="""int. number of training epochs to run""")

    parser.add_argument('--logging_steps', dest='logging_steps', default=500, required=False,
                        help="""int. size of logging steps to train on""")

    parser.add_argument('--save_steps', dest='save_steps', default=1000, required=False,
                        help="""int. size of save steps to evaluate on""")
    
    parser.add_argument('--max_length', dest='max_length', default=128, required=False,
                        help="""int. size of max length for each squence""")

    return parser.parse_args()

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    # Get access to model types and model configs to select GPT2 model and config
    MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
    MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )

    mlm: bool = field(
        default=False,
        metadata={
            "help": "Train with masked-language modeling loss instead of language modeling."
        },
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )

# Create LineByLineDataset from Movie Plots text file
def get_dataset(
    args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size
        )
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache,
        )

def main():
    # Check CPU/GPU device
    print("Check if GPU available", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("If CPU or GPU Selected", device)

    if device == "cuda:0":
        def print_gpu_utilization():
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU memory occupied: {info.used//1024**2} MB.")

        def print_summary(result):
            print(f"Time: {result.metrics['train_runtime']:.2f}")
            print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
            print_gpu_utilization()

        print("====Beginning GPU Utilization===")
        print_gpu_utilization()

    # Import Args
    args = parse_parameters()
    input_filename_train = args.input_filename_train
    input_filename_test = args.input_filename_test
    model_name = args.model_name
    output_model_path = args.output_model_path
    num_train_epochs = int(args.num_train_epochs)
    logging_steps = int(args.logging_steps)
    save_steps = int(args.save_steps)
    max_length = int(args.max_length)

    model_args = ModelArguments(
        model_name_or_path=model_name, model_type=model_name
    )
    data_args = DataTrainingArguments(
        train_data_file=input_filename_train,
        eval_data_file=input_filename_test,
        line_by_line=True,
        block_size=max_length,
        overwrite_cache=True,
    )
    training_args = TrainingArguments(
        output_dir=output_model_path+model_name,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        logging_steps=logging_steps,
        per_device_train_batch_size=4,
        num_train_epochs=num_train_epochs,
        save_total_limit=1,
        save_steps=save_steps,
    )

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logger = logging.getLogger(__name__)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed for deterministic training runs
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )

    if model_name == "gpt2" or model_name == "distilgpt2":
        model = GPT2LMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        ).to(device)
    else:
        model = BertLMHeadModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        ).to(device)

    special_tokens_dict = {
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",
        "additional_special_tokens": [
            "<superhero>",
            "<action>",
            "<drama>",
            "<thriller>",
            "<horror>",
            "<sci_fi>",
        ],
    }

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if data_args.block_size <= 0: 
      # If block_size <= 0, set it to max. possible value allowed by model
        data_args.block_size = tokenizer.model_max_length
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Get datasets

    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
        if training_args.do_eval
        else None
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=data_args.mlm,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Training
    try:
      if training_args.do_train:
          model_path = (
              model_args.model_name_or_path
              if model_args.model_name_or_path is not None
              and os.path.isdir(model_args.model_name_or_path)
              else None
          )
          trainer.train(model_path=model_path)
          trainer.save_model()
          tokenizer.save_pretrained(training_args.output_dir)
    except KeyboardInterrupt:
      print("Saving model that was in the middle of training")
      trainer.save_model()
      tokenizer.save_pretrained(training_args.output_dir)
      return

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
                    print("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    if device == "cuda:0":
        print("====Ending GPU Utilization===")
        print_gpu_utilization()      

    return results
   
if __name__ == "__main__":
    main()
