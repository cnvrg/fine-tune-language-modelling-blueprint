import os
import pandas as pd
import pathlib
import argparse
import sys
import shutil

def parse_parameters():
    parser = argparse.ArgumentParser(description="""3 models comparison""")
    parser.add_argument('-model_name_1', '--model_name_1', action='store', dest='model_name_1', default='bert-base-uncased', required=False, 
                        help="""string. name for model 1""")

    parser.add_argument('-model_path_1', '--model_path_1', action='store', dest='model_path_1', default='/input/dev_ft_bert/story_generator_checkpoint_', required=False, 
                help="""string. path for model 1""")

    parser.add_argument('-model_name_2', '--model_name_2', action='store', dest='model_name_2', default='distilgpt2', required=False, 
                        help="""string. name for model 2""")

    parser.add_argument('-model_path_2', '--model_path_2', action='store', dest='model_path_2', default='/input/dev_ft_distilgpt2/story_generator_checkpoint_', required=False, 
                        help="""string. path for model 2""")
    
    parser.add_argument('-model_name_3', '--model_name_3', action='store', dest='model_name_3', default='gpt2', required=False, 
                        help="""string. name for model 3""")

    parser.add_argument('-model_path_3', '--model_path_3', action='store', dest='model_path_3', default='/input/dev_ft_gpt/story_generator_checkpoint_', required=False, 
                        help="""string. path for model 3""")
    return parser.parse_args()

def evaluation(filelist, filesList):
    eval_df = pd.DataFrame()
    for index, value in enumerate(filelist):
        print('index', index)
        frame = pd.read_csv(filesList[index], sep="=", header=None, index_col=False, names=['col', 'perplexity'])
        eval_df = eval_df.append(frame)
    eval_df.reset_index(inplace = True)
    eval_df.drop(['col', 'index'], axis=1, inplace=True)
    eval_df.insert(0, 'model', filelist)
    print(eval_df)
    
    winner_index = eval_df[['perplexity']].idxmin()[0]
    winner = eval_df.iloc[eval_df[['perplexity']].idxmin()]['model'][0]
    print("winner model is: ", winner)
    return winner, winner_index

def compare():
    # cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
    scripts_dir = pathlib.Path(__file__).parent.resolve()
    sys.path.append(str(scripts_dir))
    cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", scripts_dir)
    
    print('Current Path', os.getcwd())

    args = parse_parameters()
    model_name_1 = args.model_name_1
    model_name_2 = args.model_name_2
    model_name_3 = args.model_name_3
    filelist = [model_name_1, model_name_2, model_name_3]

    model_path_1 = args.model_path_1
    model_path_2 = args.model_path_2
    model_path_3 = args.model_path_3
    filesList = [model_path_1+model_name_1+'/eval_results_lm.txt', 
                 model_path_2+model_name_2+'/eval_results_lm.txt', 
                 model_path_3+model_name_3+'/eval_results_lm.txt']
    pathList = [model_path_1+model_name_1, 
                 model_path_2+model_name_2, 
                 model_path_3+model_name_3]

    winner, winner_index = evaluation(filelist, filesList)
    shutil.move(pathList[winner_index], cnvrg_workdir)

if __name__ == "__main__":
    compare()