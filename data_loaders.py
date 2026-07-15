import pandas as pd
import numpy as np
import ast
from pathlib import Path
from preprocess_data import load_dataset, preprocess_dataset, add_metaData, export_mean_data, export_data, \
    assign_dataSplit, load_dataset_frames, assign_clusterInfo
from utils import string_to_bool_array



def load_humanTrialData(base_dir= f'/braintree/data2/active/users/aicha/Ego4D_data',
                        dataset_root_path = "/braintree/data2/active/users/aicha/Ego4D_data",
                        experiment_id='SensoryHistory_main', study_ID='Study1', recompute=False,
                        file_name='Data_cleaned_annotated_rebalanced.csv', save_worker_replacement=False):

    #file_path = f'{base_dir}/{experiment_id}/Export/{file_name}'
    file_path = f'{base_dir}/{experiment_id}/{file_name}'
    if Path(file_path).is_file() & (recompute == False) :
        df = pd.read_csv(file_path)
        df['final_choice'] = df['final_choice'].apply(string_to_bool_array)
        df['categories'] = df['categories'].apply(ast.literal_eval)
    else:
        data_dir = Path(f'{base_dir}/{experiment_id}/Data/{study_ID}/')
        df_raw = load_dataset(data_dir=data_dir, save_file='Dataset_rebalanced.pkl',recompute=recompute, save=True)
        df, worker_to_replace = preprocess_dataset(df_raw)

        if save_worker_replacement:
            worker_to_replace.to_csv(f'{base_dir}/{experiment_id}/version_workers_to_replace.csv'
                                     )

        df = add_metaData(df, base_dir=base_dir)

        df = assign_dataSplit(df, dataset_root_path=dataset_root_path)

        df = assign_clusterInfo(df, dataset_root_path=dataset_root_path, suffix='_humanReports_rebalanced', )
        print('Exporting...')
        export_mean_data(df, base_dir=base_dir, experiment_id=experiment_id, suffix='_humanReports_rebalanced',
                        )

        export_data(df, base_dir=base_dir, experiment_id=experiment_id,
                    file_name=file_name)

    return df


def load_humanTrialData_frames(file_name='Data_cleaned_annotated.csv', experiment_id= 'SensoryHistory_main_targetFrame_baseline',
        base_dir=f'/braintree/data2/active/users/aicha/Ego4D_data/ContinuousRecognition', recompute=False,study_ID='Study1',):
    file_path = f'{base_dir}/{experiment_id}/Export/{file_name}'
    if Path(file_path).is_file() & (recompute == False):
        df = pd.read_csv(file_path)
        df['final_choice'] = df['final_choice'].apply(string_to_bool_array)
        df['categories'] = df['categories'].apply(ast.literal_eval)
    else:
        data_dir = Path(f'{base_dir}/{experiment_id}/Data/{study_ID}/')
        df_raw = load_dataset_frames(data_dir=data_dir, save_file='Dataset.pkl')  # ,recompute=recompute, save=True)
        df, worker_to_replace = preprocess_dataset(df_raw, group_variable='image_type')

        # exp_trial_definition = pd.read_csv(
        #     f'/Users/lynn/Dropbox (MIT)/PycharmProjects/ContinuousRecognition/SensoryHistory_main/ExperimentalVideos/test_dSH-info.csv')
        # exp_trial_definition['Filename'] = exp_trial_definition['Filename'].str[:-4]
        # for var in ['duration_clip_sec', 'startTime_clip_sec']:
        #     df_exp.loc[df['image_type'] == 'FinalFrame', var] = df_exp.loc[
        #         df_exp['image_type'] == 'FinalFrame', 'File_name'].map(
        #         dict(zip(exp_trial_definition['Filename'].tolist(),
        #                  exp_trial_definition[var].tolist())))
        #
        # df_exp.rename(columns={'duration_clip_sec': 'videoDuration (sec)',
        #                        'startTime_clip_sec': 'startTime_clip (sec)'}, inplace=True)
        export_data(df, base_dir=base_dir, experiment_id=experiment_id,
                    file_name=file_name)

    return df

def load_humanReportRates(dataset, base_dir=f'/braintree/data2/active/users/aicha/Ego4D_data/ContinuousRecognition/',
                        dataset_root_path='/braintree/data2/active/users/aicha/Ego4D_data',
                        experiment_id='SensoryHistory_main', study_ID='Study1', recompute=False,
                        suffix='_humanReports_rebalanced'):
    file_path = f'{base_dir}/{experiment_id}/Export/{dataset}{suffix}.csv'
    if (Path(file_path).is_file() == False) | (recompute == True):
        df = load_humanTrialData(base_dir=base_dir, dataset_root_path=dataset_root_path, experiment_id=experiment_id,
                            study_ID=study_ID, recompute=recompute)
    else:
        df = pd.read_csv(file_path)
    return df