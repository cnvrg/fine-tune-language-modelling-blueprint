# language Modelling Blueprint Model Comparison
## _cnvrg_

The Language Modelling Compare library selects the best-performing model based on the perplexity. The selected model can then be used for subsequent tasks such as inference using an endpoint.

Click [here](https://github.com/cnvrg/fine-tune-language-modelling-blueprint) for more information on this library.

## Library Flow
The following list outlines this library's high-level flow:
- The library iterates through all the models artifacts and chooses the model with the lowest perplexity.
- It moves the saved model and other artifacts to from previous libraries to the current working directory.

## Inputs
This library mainly operates on artifacts from previous libraries with following input arguments.

## Parameters

`--model_name_1` - (String) (Default: 'bert-base-uncased') Name of the model.

`--model_path_1` - (String) (Default: '/input/dev_ft_bert/story_generator_checkpoint_') Path to bert model's checkpoint.

`--model_name_2` - (String) (Default: 'distilgpt22') Name of the model.

`--model_path_2` - (String) (Default: '/input/dev_ft_bert/story_generator_checkpoint_') Path to distilgpt2 model's checkpoint.

`--model_name_3` - (String) (Default: 'gpt2') Name of the model.

`--model_path_3` - (String) (Default: '/input/dev_ft_bert/story_generator_checkpoint_') Path to gpt2 model's checkpoint.

## Sample Command
Refer to the following sample command:

```bash
python compare.py
```

## Outputs
The Model Compare library does not generate any output artifacts by itself. It moves artifacts from previous libraries to the working directory `/cnvrg`.

## Troubleshooting
- Check the experiment's Artifacts section to confirm the library has moved the required files.