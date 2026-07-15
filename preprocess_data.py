from pathlib import  Path
import os
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from load_trialTypes import loadExpTrials, loadBenchmarkTrials, loadVisualizationTrials, loadClipSeqTrials
from utils import compute_meanHR

cm = 1 / 2.54


def load_dataset(data_dir=None, recompute=False, save_file='Dataset.pkl', flag='qualified',  save=False, include_videoID=True):
    if data_dir == None:
        data_dir = Path(__file__).parent.parent / 'Data' / 'Experiment1'

    if (recompute == False) & (data_dir / save_file).exists():
        print(f'Loading assembled dataset from {data_dir / save_file}')
        df = pd.read_pickle(data_dir / save_file)

    else:
        print(f'Reassembling dataset.')
        data_files = os.listdir(data_dir)
        data_files = [file for file in data_files if file.endswith(f'{flag}.json')]
        print(len(data_files))
        df = []
        for data_file in data_files:
            with open(data_dir / data_file, 'r') as file:
                data = json.load(file)

            HIT_id, worker_id, assignment_id, version, flag = data_file.split('_')

            assignment_id = assignment_id.split('.')[0]


            for i, trial in enumerate(data):
                if 'trial_outcome' in trial:

                    current_data = trial['trial_outcome']

                    current_df = {'HIT_ID': HIT_id,
                                  'worker_ID': worker_id,
                                  'assignment_ID': assignment_id,

                                  'Trial_number': current_data['trial_number'],

                                   'Video_ID': current_data['video_ID'] if include_videoID else None,

                                  'stimulus_video_url': current_data['stimulus_video_url'],

                                  'video_type': current_data['video_type'],

                                  'version': version,
                                  'label': current_data['label'],
                                  'n_choices': sum(current_data['final_choice']),
                                  'final_choice': np.array(current_data['final_choice']),

                                  'timestamp_start': current_data['timestamp_start'],
                                  'categories': current_data['categories'],

                                  'practice': current_data['practice'],
                                  'correct': current_data['perf'],
                                  'bonus?': current_data['bonus_usd_if_correct'],
                                  'button_rt': current_data['button_rt'],

                                  }
                    final_categories = np.array(current_data['categories'])[np.array(current_data['final_choice'])]
                    choice_count = 0
                    for c in range(len(current_data['choices_made'])):
                        category_choice = current_data['choices_made'][c].split('/')[-1].split('.')[0][1:]
                        if category_choice in final_categories:  # Ignore those that are not part of the final selection
                            current_df[f'choice{choice_count}'] = category_choice
                            current_df[f'choice{choice_count}_rt'] = current_data['reaction_time_msec'][c]
                            choice_count = choice_count + 1

                    df.append(current_df)


        df = pd.DataFrame(df)
        df['File_ID'] = df['video_type'] + '_' + df['Video_ID']
        df['File_name'] = df['stimulus_video_url'].apply(lambda x: x.split('/')[-1][:-4])
        df['File_ID'] = df['video_type'] + '_' + df['File_name']
        df['video_group'] = df['video_type'].str.split('_', expand=True)[0]
        if save:
            df.to_pickle(data_dir / save_file)
    return df

def load_dataset_frames(data_dir=None, recompute=False, save_file='Dataset.pkl', flag='qualified',  save=False):
    if data_dir == None:
        data_dir = Path(__file__).parent.parent / 'Data' / 'Experiment1'

    if (recompute == False) & (data_dir / save_file).exists():
        print(f'Loading assembled dataset from {data_dir / save_file}')
        df = pd.read_pickle(data_dir / save_file)

    else:
        print(f'Reassembling dataset.')
        data_files = os.listdir(data_dir)
        data_files = [file for file in data_files if file.endswith(f'{flag}.json')]
        print(len(data_files))
        df = []
        for data_file in data_files:
            with open(data_dir / data_file, 'r') as file:
                data = json.load(file)

            HIT_id, worker_id, assignment_id, version, flag = data_file.split('_')

            assignment_id = assignment_id.split('.')[0]


            for i, trial in enumerate(data):
                if 'trial_outcome' in trial:
                    current_data = trial['trial_outcome']

                    current_df = {'HIT_ID': HIT_id,
                                  'worker_ID': worker_id,
                                  'assignment_ID': assignment_id,

                                  'Trial_number': current_data['trial_number'],

                                   'Video_ID': current_data['video_ID'],

                                  'image_url': current_data['image_url'],
                                  'image_type': current_data['image_type'],

                                  'version': version,
                                  'label': current_data['label'],
                                  'n_choices': sum(current_data['final_choice']),
                                  'final_choice': np.array(current_data['final_choice']),

                                  'timestamp_start': current_data['timestamp_start'],
                                  'categories': current_data['categories'],

                                  'practice': current_data['practice'],
                                  'correct': current_data['perf'],
                                  'bonus?': current_data['bonus_usd_if_correct'],
                                  'button_rt': current_data['button_rt'],

                                  }
                    final_categories = np.array(current_data['categories'])[np.array(current_data['final_choice'])]
                    choice_count = 0
                    for c in range(len(current_data['choices_made'])):
                        category_choice = current_data['choices_made'][c].split('/')[-1].split('.')[0][1:]
                        if category_choice in final_categories:  # Ignore those that are not part of the final selection
                            current_df[f'choice{choice_count}'] = category_choice
                            current_df[f'choice{choice_count}_rt'] = current_data['reaction_time_msec'][c]
                            choice_count = choice_count + 1

                    df.append(current_df)


        df = pd.DataFrame(df)
        df['File_ID'] = df['image_type'] + '_' + df['Video_ID']
        df['File_name'] = df['image_url'].apply(lambda x: x.split('/')[-1][:-4])
        df['File_ID'] = df['image_type'] + '_' + df['File_name']
        #df['video_group'] = df['video_type'].str.split('_', expand=True)[0]
        if save:
            df.to_pickle(data_dir / save_file)
    return df


def load_meta(data_dir=None, recompute=False, save_file='Comments.pkl', save=False):
    if data_dir == None:
        data_dir = Path(__file__).parent.parent / 'Data' / 'Experiment1'

    if (recompute == False) & (data_dir / save_file).exists():
        print(f'Loading assembled dataset from {data_dir / save_file}')
        df = pd.read_pickle(data_dir / save_file)

    else:
        print(f'Reassembling dataset.')
        data_files = os.listdir(data_dir)
        data_files = [file for file in data_files if file.endswith('qualified.json')]
        print(len(data_files))
        df = []
        for data_file in data_files:
            with open(data_dir / data_file, 'r') as file:
                data = json.load(file)

            HIT_id, worker_id, assignment_id, version, flag = data_file.split('_')

            assignment_id = assignment_id.split('.')[0]


            for i, trial in enumerate(data):
                if trial['trial_type'] == 'survey-text':

                    current_df = {'HIT_ID': HIT_id,
                                  'worker_ID': worker_id,
                                  'assignment_ID': assignment_id,

                                  'Comment': trial['response']['Q0'],


                                  }

                    df.append(current_df)
            # Compute duration
            df[-1]['ExperimentDuration_sec'] = (data[-1]['time_elapsed'] - data[0]['time_elapsed'])/1000

        df = pd.DataFrame(df)
        if save:
            df.to_pickle(data_dir / save_file)
    return df



def preprocess_dataset(df, plot=False, include_catch=False, group_variable='video_group', n_assignments=9):

    # Define to be excluded trial-level behaviors
    # 1. selected all categories
    # 2. click 0 categories and be very fast (< 0.8s?) or very slow (> 7.5s)
    # 3. Take a very long time given the number of categories selected (> 97.5 quantile).
    if include_catch ==True:
        df = df.loc[~df[group_variable].isin(['practice'])]
    else:
        df= df.loc[~df[group_variable].isin(['practice', 'Catch'])]


    filter_idx = []
    filter_idx.extend(df.loc[(df['n_choices'] == 12)].index)
    filter_idx.extend(df.loc[(df['n_choices'] == 0) & ((df['button_rt'] < 800) | (df['button_rt'] > 7500))].index)

    for n in range(1, 12):
        upper_bound = df.loc[(df['n_choices'] == n)]['button_rt'].quantile(0.975)
        filter_idx.extend(df.loc[(df['n_choices'] == n) & (df['button_rt'] > upper_bound)].index)

    # Look at subject level oddities
    # Are currently excluded trials disproportionately affecting specific workers?
    filtered_df = df.loc[(~df.index.isin(filter_idx))]
    total_eligible_trials = len(df.loc[(~df[group_variable].isin(['practice', 'Catch']))])
    print(f'Discarded {len(filter_idx)} of {total_eligible_trials} eligible trials ({len(filter_idx)/total_eligible_trials*100:.2f}%)')

    proportion_excludedTrials_worker = 1 - filtered_df.groupby('worker_ID').size()/df.groupby('worker_ID').size() # 100 trials per exp
    if plot:
        g = sns.displot(data=proportion_excludedTrials_worker.reset_index(), x=0, height=6*cm)
        g.set_xlabels('Excluded proportion of trials per worker')
        g.set_ylabels('# workers')
        plt.show()

    excluded_workers = proportion_excludedTrials_worker[proportion_excludedTrials_worker > 0.2].index.tolist()

    # Is a worker always giving the same number of responses and using the same categories?
    variety_choices_workers = filtered_df.groupby('worker_ID')['n_choices'].nunique()
    if plot:
        g = sns.displot(data=variety_choices_workers.reset_index(), x='n_choices', discrete=True, height=6*cm)
        g.set_xlabels('Types of choices')
        g.set_ylabels('# workers')

        plt.show()

    excluded_workers.extend(variety_choices_workers[variety_choices_workers == 1].index.tolist()) # exclude those that always chose the same number of categories (likely 1) as this indicates that they didn't understand the task instructions.
    total_eligible_workers = filtered_df['worker_ID'].nunique()
    print(
        f'Excluded {len(excluded_workers)} of {total_eligible_workers} eligible workers ({len(excluded_workers) / total_eligible_workers * 100:.2f}%)')

    version_workers_to_replace = n_assignments - filtered_df.loc[~filtered_df['worker_ID'].isin(excluded_workers)].groupby('version')['worker_ID'].nunique()
    version_workers_to_replace = version_workers_to_replace[version_workers_to_replace > 0]
    filtered_df = filtered_df.loc[~filtered_df['worker_ID'].isin(excluded_workers)]
    print(
        f'Final: {len(filtered_df)} of {total_eligible_trials} eligible trials ({len(filtered_df) / total_eligible_trials * 100:.2f}%)')

    filtered_df = filtered_df.reset_index()
    return filtered_df, version_workers_to_replace

def add_metaData(df, base_dir=None, experiment_id = 'SensoryHistory_main',
                 datasets=['train_SH', 'train_dSH', 'test_SH',  'test_dSH', 'visualization', 'benchmark', 'clipSequences'],
                 vars = ['videoDuration (sec)']):
    if base_dir is None:
        base_dir = f'/braintree/data2/active/users/aicha/Ego4D_data/ContinuousRecognition/'
    target_dir = f'{base_dir}{experiment_id}'
    benchmark_dir = f"{base_dir}/Benchmarks/"
    # load trial defs
    ### TRIAL SAMPLING
    exp_trial_definition = loadExpTrials(target_dir)
    exp_trial_definition_dynamic = loadExpTrials(target_dir, datasets=['train_dSH', 'test_dSH'])
    exp_trial_definition = pd.concat([exp_trial_definition, exp_trial_definition_dynamic], ignore_index=True)
    benchmark_trial_definition = loadBenchmarkTrials(benchmark_dir)
    seq_trial_definition = loadClipSeqTrials(target_dir)
    visualization_trial_definition = loadVisualizationTrials(benchmark_dir)

    for set in datasets:
        for var in vars:

            if set == 'benchmark':
                trial_def = benchmark_trial_definition
            elif set == 'visualization':
                trial_def = visualization_trial_definition
            elif set == 'clipSequences':
                trial_def = seq_trial_definition
            else:
                trial_def = exp_trial_definition.loc[exp_trial_definition['videoType'] == set]

            df.loc[df['video_type'].str.startswith(set), var] = df.loc[
                    df['video_type'].str.startswith(set), 'stimulus_video_url'].map(
                    dict(zip(trial_def['aws_link'].tolist(),
                         trial_def[var].tolist())))

    return df


def assign_dataSplit(df, dataset_root_path='/braintree/data2/active/users/lynnka/Datasets/Ego4D_videos'):
    mapping_df_SH = pd.read_csv(f"{dataset_root_path}/Dataset_splits_TestSet_expansion.csv")

    print('Datatset splits BEFORE:\n')
    print(df.groupby(['video_group'])['File_ID'].nunique())

    for split in mapping_df_SH['Split'].unique():
        # update video group and video type
        current_map = mapping_df_SH.loc[mapping_df_SH['Split'] == split]

        df.loc[df['stimulus_video_url'].isin(current_map['stimulus_video_url'].tolist()), 'video_group'] = split

        df.loc[df['stimulus_video_url'].isin(current_map['stimulus_video_url'].tolist()), 'video_type']  = 'val_SH' if split =='val' else split + '_SH'
        df.loc[df['stimulus_video_url'].isin(current_map['stimulus_video_url'].tolist()), 'File_ID'] = df.loc[df['stimulus_video_url'].isin(current_map['stimulus_video_url'].tolist()), 'video_type'] + '_'+ df.loc[df['stimulus_video_url'].isin(current_map['stimulus_video_url'].tolist()), 'File_name']

    # Assign some train_dSH videos to the validation split
    val_dSH_urls = df.loc[(df['video_type'] == 'train_dSH')]
    val_dSH_urls = val_dSH_urls.loc[val_dSH_urls['File_name'].str.split('_', expand=True)[0].astype(int) >= 1800, 'stimulus_video_url'].unique()

    df.loc[df['stimulus_video_url'].isin(val_dSH_urls), 'video_group'] = 'val'
    df.loc[df['stimulus_video_url'].isin(val_dSH_urls), 'video_type'] = 'val_dSH'
    df.loc[df['stimulus_video_url'].isin(val_dSH_urls), 'File_ID'] = df.loc[df['stimulus_video_url'].isin(val_dSH_urls), 'video_type'] + '_'+ df.loc[df['stimulus_video_url'].isin(val_dSH_urls), 'File_name']

    df.reset_index(drop=True, inplace=True)
    # Finally, double-check that there is no overlap with the test set
    overlap_ids = df.loc[
        df['video_group'].isin(['train', 'val']) &
        df['Video_ID'].isin(df.loc[df['video_group'] == 'test', 'Video_ID'])
        , 'Video_ID'].unique()

    print(f'Found overlapping Video IDs with the test set: {len(overlap_ids)}') if len(overlap_ids) > 0 else None

    df = df[
        ~((df['video_group'].isin(['train', 'val'])) & df['Video_ID'].isin(overlap_ids))
    ]


    print('Datatset splits AFTER:\n')
    print(df.groupby(['video_group'])['File_ID'].nunique())

    return df

def assign_clusterInfo(df, datasets=None, dataset_root_path='/braintree/data2/active/users/lynnka/Datasets/Ego4D_videos',
                      suffix='_humanReports_rebalanced',):

    if datasets is None:
        datasets = ['train', 'val', 'test']

    cluster_info_df = []
    for dataset in datasets:
        cluster_info_df.append(pd.read_csv(f'{dataset_root_path}/clusterAnnotations_{dataset}{suffix}.csv'))

    cluster_info_df = pd.concat(cluster_info_df, ignore_index=True)

    for variable in ['n_clusters', 'n_noise_points','cluster_per_frame']:
        df[variable] = df['stimulus_video_url'].map(dict(zip(cluster_info_df['stimulus_video_url'], cluster_info_df[variable])))

    return df

def export_mean_data(df, base_dir=None, experiment_id = 'SensoryHistory_main', suffix='', include_catch=False):

    categories = df.loc[0, 'categories']
    train_df = \
    df.loc[df['video_group'] == 'train'].groupby(['File_name', 'File_ID', 'stimulus_video_url', 'Video_ID', 'video_type', 'videoDuration (sec)', 'n_clusters'],
                                                 as_index=False)['final_choice'].apply(lambda x: compute_meanHR(x))

    train_df['N_reports'] = df.loc[df['video_group'] == 'train'].groupby(['File_name', 'File_ID', 'stimulus_video_url', 'Video_ID', 'video_type', 'videoDuration (sec)'])['worker_ID'].nunique().tolist()

    train_df['video_number'] = train_df['File_name'].str.split('_').str[0].astype(int)
    train_df.sort_values(by=['video_number'], inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    train_df[categories] = train_df['final_choice'].apply(pd.Series)

    # g = sns.displot(data=test_df, x='N_reports', discrete = True, hue='video_type')
    # plt.show()
    # Create validation set
    val_df = \
        df.loc[df['video_group'] == 'val'].groupby(
            [ 'File_name', 'File_ID','stimulus_video_url', 'Video_ID', 'video_type', 'videoDuration (sec)', 'n_clusters'],
            as_index=False)['final_choice'].apply(lambda x: compute_meanHR(x))

    val_df['N_reports'] = df.loc[df['video_group'] == 'val'].groupby(
        ['File_name', 'File_ID', 'stimulus_video_url', 'Video_ID', 'video_type', 'videoDuration (sec)'])[
        'worker_ID'].nunique().tolist()

    val_df['video_number'] = train_df['File_name'].str.split('_').str[0].astype(int)
    val_df.sort_values(by=['video_number'], inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    val_df[categories] = val_df['final_choice'].apply(pd.Series)

    test_df = \
    df.loc[df['video_group'] == 'test'].groupby(['File_name', 'File_ID', 'stimulus_video_url', 'Video_ID', 'video_type', 'videoDuration (sec)', 'n_clusters'],
                                                as_index=False)['final_choice'].apply(lambda x: compute_meanHR(x))

    test_df['N_reports'] = df.loc[df['video_group'] == 'test'].groupby(
        ['File_name', 'File_ID', 'stimulus_video_url', 'Video_ID', 'video_type', 'videoDuration (sec)'])[
        'worker_ID'].nunique().tolist()

    test_df['video_number'] = test_df['File_name'].str.split('_').str[0].astype(int)
    test_df.sort_values(by=['video_number'], inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    test_df[categories] = test_df['final_choice'].apply(pd.Series)


    vis_df = df.loc[df['video_group'] == 'visualization'].groupby(['video_group', 'File_name', 'File_ID', 'stimulus_video_url', 'Video_ID', 'video_type'],
                                                as_index=False)['final_choice'].apply(lambda x: compute_meanHR(x))

    vis_df['N_reports'] = df.loc[df['video_group'] == 'visualization'].groupby(
        ['video_group', 'File_name', 'File_ID', 'stimulus_video_url', 'Video_ID', 'video_type'])[
        'worker_ID'].nunique().tolist()

    vis_df['videoDuration (sec)'] = vis_df['video_type'].str.split('_').str[1].str[:-1].astype(float)
    vis_df.reset_index(drop=True, inplace=True)
    vis_df[categories] = vis_df['final_choice'].apply(pd.Series)

    bench_df = df.loc[df['video_group'] == 'benchmark'].groupby(
        ['video_group', 'File_name', 'File_ID', 'stimulus_video_url', 'Video_ID', 'video_type', 'videoDuration (sec)'],
        as_index=False)['final_choice'].apply(lambda x: compute_meanHR(x))

    bench_df['N_reports'] = df.loc[df['video_group'] == 'benchmark'].groupby(
        ['video_group', 'File_name', 'File_ID', 'stimulus_video_url', 'Video_ID', 'video_type', 'videoDuration (sec)'])[
        'worker_ID'].nunique().tolist()

    bench_df['Benchmark_condition'] = bench_df['stimulus_video_url'].str.split('/', expand=True)[5]
    bench_df.loc[bench_df['Benchmark_condition'] != 'Baseline', 'Benchmark_subcondition'] = bench_df.loc[bench_df['Benchmark_condition'] != 'Baseline', 'stimulus_video_url'].str.split('/', expand=True)[6]

    bench_df.reset_index(drop=True, inplace=True)
    bench_df[categories] = bench_df['final_choice'].apply(pd.Series)

    if include_catch:
        catch_df = df.loc[df['video_group'] == 'Catch'].groupby(
            ['video_group', 'File_name', 'stimulus_video_url', 'Video_ID', 'video_type'],
            as_index=False)['final_choice'].apply(lambda x: compute_meanHR(x))

        catch_df.reset_index(drop=True, inplace=True)

        catch_df[categories] = catch_df['final_choice'].apply(pd.Series)
        catch_df.to_csv(f'{base_dir}{experiment_id}/Export/catch{suffix}.csv', index=False)

    train_df.to_csv(f'{base_dir}{experiment_id}/Export/train{suffix}.csv', index=False)
    val_df.to_csv(f'{base_dir}{experiment_id}/Export/val{suffix}.csv', index=False)
    test_df.to_csv(f'{base_dir}{experiment_id}/Export/test{suffix}.csv', index=False)

    vis_df.to_csv(f'{base_dir}{experiment_id}/Export/visualization{suffix}.csv', index=False)
    bench_df.to_csv(f'{base_dir}{experiment_id}/Export/benchmarks{suffix}.csv', index=False)



def export_data(df, base_dir=None, experiment_id = 'SensoryHistory_main', file_name='Data_cleaned_annotated.csv'):

    df.to_csv(f'{base_dir}{experiment_id}/Export/{file_name}', index=False)