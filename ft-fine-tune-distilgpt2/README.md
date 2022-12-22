#  cnvrg fine tune pre-trained model for language modelling to design chatbot

Notes for this Component - 

## How It Works

The library trains a distilgpts model for 30K stories on different subjects and produces a language model and a tokenizer.
By default the library needs the receive a single path (--input_filename) for a local file.
The library performs fine-tuning of the pre-trained distilgpt2 model from huggingface with its associated tokenizer.   


## How To Run

python3 fine-tune.py

run python3 fine-tune.py -f  for info about more optional parameters of hyper parameters.
                                     
## Parameters

`--input_filename_train` - (String) (Required param) Path to a local stories data file for train.

`--input_filename_test` - (String) (Required param) Path to a local stories data file for test.

`--model_name` - (String) (Default: 'distilgpt2') Name of the model.

`--output_model_path` - (String) (Default: '/cnvrg/story_generator_checkpoint_') Path to save model checkpoint and tokenizer.

`--num_train_epochs` - (int) (Default: 3) The number of epochs the algorithm performs in the training phase.

`--logging_steps` - (int) (Default: 5000) The number of texts the model goes over in each epoch for training phase.

`--save_steps` - (int) (Default: 1000) The number of texts the model goes over in each epoch for evaluation phase.

`--max_length` - (int) (Default: 256) The number of characters per sequence.