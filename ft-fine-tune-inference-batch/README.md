#  cnvrg batch prediction by using fine tune pre-trained model for language modeling

Notes for this Component - 

## How It Works

The library performs batch prediction by using the best fine-tuned pre-trained model for language modeling on 30K movie stories based on different subjects.
By default the library needs the receive a single path (--input_filename) for a local file with dataset or for a dataset from S3 storage, a model path (--best_model_path or --model_path) with the model checkpoint, tokenizer, optimizer and etc. The library performs batch prediction by using specified model and tokenizer.  

## How To Run

python3 inference-batch.py 

run python3 fine-tune-inference.py -f  for info about more optional parameters
                                     
## Parameters

`--input_filename` - (String) (Required param) Path to a local labeled data file which contains the data that is used for prediction.

`--model_name` - (String) (Default: 'None') Name of the model that user selects.

`--model_path` - (String) (Default: './story_generator_checkpoint_') Path to load the saved model's checkpoint and tokenizer.

`--best_model_path` - (String) (Default: '/input/ft_compare_language/story_generator_checkpoint_') Path to load the automatically saved best model's checkpoint and tokenizer.

`--result_path` - (String) (Default: '/cnvrg') Path for saving the result.

`--text_column` - (String) (Default: 'text') Name of text column in dataframe.

`--length_story` - (int) (Default: 100) The number of characters per sequence to generate story.


