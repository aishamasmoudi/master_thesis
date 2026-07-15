import pandas as pd
import numpy as np
from pathlib import Path

import os
### MAIN DATASET TRIALS
def loadExpTrials(target_dir, datasets = ['train_SH', 'test_SH']):
    exp_trial_definition = []
    for dataset in datasets:
        df = pd.read_csv(f'{target_dir}/ExperimentalVideos/{dataset}-info.csv')
        for c in df['Count'].unique():
            video_id = df.loc[df['Count'] == c, 'Video_ID'].item()
            exp_trial_definition.append({'Video_ID': video_id,
                                             'videoType': dataset,
                                         'File_ID': f'{c}_{video_id}',
                                             'videoDuration (sec)': df.loc[df['Count'] == c, 'duration_clip_sec'].item(),
                                     'startTime_clip (sec)': df.loc[df['Count'] == c, 'startTime_clip_sec'].item(),
                                     'Feedback_Training': 'null',
                                             'Training': 'No',
                                             'Video_file': None,
                                             'aws_link': f'https://lynnka-resources.s3.amazonaws.com/SensoryHistory_datasets/{dataset}/{c}_{video_id}.mp4',
                                             'Label': None,
                                             'i_correct_choice': 'null',
                                         'concurrent_video_set_id':  df.loc[df['Count'] == c, 'concurrent_video_set_id'].item(),
                                         'scenarios': df.loc[df['Count'] == c, 'scenarios'].item(),
                                         'baseDuration': df.loc[df['Count'] == c, 'baseDuration'].item()
                                             })
    return pd.DataFrame(exp_trial_definition)


### BENCHMARKS TRIALS
def loadBenchmarkTrials(benchmark_dir,
                        sh_conditions = [0.25, 0.5, 1.0, 2.0, 4.0],
                        es_conditions=[0.25, 2.0],
                        td_conditions = [-1, -2,  -4, -8, -15],
                        op_conditions = ['Baseline_OP', 'Disappearance', 'RelevantSquare', 'IrrelevantSquare', 'NaturalDisappearance']
                        ):
    trial_definition = pd.read_excel(f'{benchmark_dir}BaseVideoDefinition_overview.xlsx')
    # add baseline videos

    baseline_videos = trial_definition.loc[trial_definition['OtherBenchmarks'] == 'Yes', 'Target_ID'].tolist()
    benchmark_trial_definition = []
    for target_id in baseline_videos:
        video_id = trial_definition.loc[trial_definition['Target_ID'] == target_id,  'Video_ID'].item()
        video_file = f"{target_id}_{video_id}.mp4"
        benchmark_trial_definition.append({'Video_ID': video_id,
                                             'videoType': 'benchmark_baseline',
                                           'File_ID': f'baseline_{video_id}',
                                             'videoDuration (sec)': 6,
                                            'startTime_clip (sec)': trial_definition.loc[trial_definition['Target_ID'] == target_id, 'PreprocessedClip_start (sec)'].item() + trial_definition.loc[trial_definition['Target_ID'] == target_id, 'BenchmarkClip_start(sec)'].item(),
                                                'Feedback_Training': 'null',
                                             'Training': 'No',
                                             'Video_file': f'{benchmark_dir}Baseline/{video_file}',
                                             'aws_link': f'https://lynnka-resources.s3.amazonaws.com/SensoryHistory_datasets/Benchmarks/Baseline/{video_file}',
                                             'Label': None,
                                             'i_correct_choice': 'null'
                                             })

    sh_dir = f'{benchmark_dir}SensoryHistory/'
    for sh_condition in sh_conditions:
        for target_id in baseline_videos:
            video_id = trial_definition.loc[trial_definition['Target_ID'] == target_id, 'Video_ID'].item()
            video_file = f"{target_id}_{video_id}.mp4"
            benchmark_trial_definition.append({'Video_ID': video_id,
                                         'videoType': f'benchmark_SH_{sh_condition}s',
                                               'File_ID': f'SH-{sh_condition}s_{video_id}',
                                         'videoDuration (sec)': 6,
                                         'startTime_clip (sec)': trial_definition.loc[trial_definition[
                                                                                          'Target_ID'] == target_id, 'PreprocessedClip_start (sec)'].item() +
                                                                 trial_definition.loc[trial_definition[
                                                                                          'Target_ID'] == target_id, 'BenchmarkClip_start(sec)'].item(),
                                         'Feedback_Training': 'null',
                                         'Training': 'No',
                                         'Video_file': f'{sh_dir}{sh_condition}s/{video_file}',
                                         'aws_link': f'https://lynnka-resources.s3.amazonaws.com/SensoryHistory_datasets/Benchmarks/SensoryHistory/{sh_condition}s/{video_file}',
                                         'Label': None,
                                         'i_correct_choice': 'null'
                                         })

    es_dir = f'{benchmark_dir}EventSegmentation/'

    for es_condition in es_conditions:
        for target_id in baseline_videos:
            video_id = trial_definition.loc[trial_definition['Target_ID'] == target_id, 'Video_ID'].item()
            video_file = f"{target_id}_{video_id}.mp4"
            benchmark_trial_definition.append({'Video_ID': video_id,
                                         'videoType': f'benchmark_ES_{es_condition}s',
                                               'File_ID': f'ES-{es_condition}s_{video_id}',
                                         'videoDuration (sec)': 6,
                                         'startTime_clip (sec)': trial_definition.loc[trial_definition[
                                                                                          'Target_ID'] == target_id, 'PreprocessedClip_start (sec)'].item() +
                                                                 trial_definition.loc[trial_definition[
                                                                                          'Target_ID'] == target_id, 'BenchmarkClip_start(sec)'].item(),
                                         'Feedback_Training': 'null',
                                         'Training': 'No',
                                         'Video_file': f'{es_dir}{es_condition}s/{video_file}',
                                         'aws_link': f'https://lynnka-resources.s3.amazonaws.com/SensoryHistory_datasets/Benchmarks/EventSegmentation/{es_condition}s/{video_file}',
                                         'Label': None,
                                         'i_correct_choice': 'null'
                                         })

            benchmark_trial_definition.append({'Video_ID': video_id,
                                               'videoType': f'benchmark_ES_{es_condition}s-baseline',
                                               'File_ID': f'ES-{es_condition}s-baseline_{video_id}',
                                               'videoDuration (sec)': 6 - es_condition,
                                               'startTime_clip (sec)': None,
                                               'Feedback_Training': 'null',
                                               'Training': 'No',
                                               'Video_file': f'{es_dir}{es_condition}s-baseline/{video_file}',
                                               'aws_link': f'https://lynnka-resources.s3.amazonaws.com/SensoryHistory_datasets/Benchmarks/EventSegmentation/{es_condition}s-baseline/{video_file}',
                                               'Label': None,
                                               'i_correct_choice': 'null'
                                               })

    td_dir = f'{benchmark_dir}Retention/'
    td_videos = trial_definition.loc[trial_definition['TemporalDecayBenchmark'] == 'Yes', 'Target_ID'].tolist()
    for td_condition in td_conditions:
        for target_id in td_videos:
            video_id = trial_definition.loc[trial_definition['Target_ID'] == target_id, 'Video_ID'].item()
            video_file = f"{target_id}_{video_id}.mp4"
            benchmark_trial_definition.append({'Video_ID': video_id,
                                         'videoType': f'benchmark_TD_{td_condition}s',
                                               'File_ID': f'TD-{td_condition}s_{video_id}',
                                         'videoDuration (sec)': 6,
                                         'startTime_clip (sec)': trial_definition.loc[trial_definition[
                                                                                          'Target_ID'] == target_id, 'PreprocessedClip_start (sec)'].item() +
                                                                 trial_definition.loc[trial_definition[
                                                                                          'Target_ID'] == target_id, 'BenchmarkClip_start(sec)'].item(),
                                         'Feedback_Training': 'null',
                                         'Training': 'No',
                                         'Video_file': f'{td_dir}{td_condition}s/{video_file}',
                                         'aws_link': f'https://lynnka-resources.s3.amazonaws.com/SensoryHistory_datasets/Benchmarks/Retention/{td_condition}s/{video_file}',
                                         'Label': None,
                                         'i_correct_choice': 'null'
                                         })


    op_dir = f'{benchmark_dir}ObjectPermanence/'
    op_videos = trial_definition.loc[trial_definition['ObjectPermanenceBenchmark'] == 'Yes', 'Target_ID'].tolist()
    for op_condition in op_conditions:
        for target_id in op_videos:
            video_id = trial_definition.loc[trial_definition['Target_ID'] == target_id, 'Video_ID'].item()
            video_file = f"{target_id}_{video_id}.mp4"
            video_path = f'{op_dir}{op_condition}/{video_file}'
            if Path(video_path).exists():
                # video = VideoFileClip(video_path)
                # video_duration = video.duration
                benchmark_trial_definition.append({'Video_ID': video_id,
                                             'videoType': f'benchmark_OP_{op_condition}',
                                                   'File_ID': f'OP-{op_condition}_{video_id}',
                                             #'videoDuration (sec)': video_duration,
                                             'startTime_clip (sec)': trial_definition.loc[trial_definition[
                                                                                              'Target_ID'] == target_id, 'PreprocessedClip_start (sec)'].item() +
                                                                     trial_definition.loc[trial_definition[
                                                                                              'Target_ID'] == target_id, 'BenchmarkClip_start(sec)'].item(),
                                             'Feedback_Training': 'null',
                                             'Training': 'No',
                                             'Video_file': f'{op_dir}{op_condition}/{video_file}',
                                             'aws_link': f'https://lynnka-resources.s3.amazonaws.com/SensoryHistory_datasets/Benchmarks/ObjectPermanence/{op_condition}/{video_file}',
                                             'Label': None,
                                             'i_correct_choice': 'null'
                                             })

    return pd.DataFrame(benchmark_trial_definition)

def loadVisualizationTrials(benchmark_dir,
                        vis_conditions =np.linspace(0.1, 4, 40).round(1)
                        ):
    trial_definition = pd.read_excel(f'{benchmark_dir}BaseVideoDefinition_overview.xlsx')
    # add baseline videos
    baseline_videos = trial_definition.loc[trial_definition['OtherBenchmarks'] == 'Yes', 'Target_ID'].tolist()
    visualization_trial_definition = []

    vis_dir = f'{benchmark_dir}Visualization/'
    vis_videos = trial_definition.loc[trial_definition['VisualizationExamples'] == 'Yes', 'Target_ID'].tolist()
    for vis_condition in vis_conditions:
        for target_id in vis_videos:
            if (target_id not in ['Bike11', 'Chair1', 'Car2']) & (vis_condition not in np.linspace(0.2, 4, 20).round(1)):

                pass
            else:
                video_id = trial_definition.loc[trial_definition['Target_ID'] == target_id, 'Video_ID'].item()
                video_file = f"{target_id}_{video_id}.mp4"
                visualization_trial_definition.append({'Video_ID': video_id,
                                                   'videoType': f'visualization_{vis_condition}s',
                                                   'File_ID': f'visualization_{vis_condition}s_{target_id}_{video_id}',
                                                   'videoDuration (sec)': 4,
                                                   'startTime_clip (sec)': trial_definition.loc[trial_definition[
                                                                                                    'Target_ID'] == target_id, 'PreprocessedClip_start (sec)'].item() +
                                                                           trial_definition.loc[trial_definition[
                                                                                                    'Target_ID'] == target_id, 'VisualizationClip_start(sec)'].item(),
                                                   'Feedback_Training': 'null',
                                                   'Training': 'No',
                                                   'Video_file': f'{vis_dir}{vis_condition}s/{video_file}',
                                                   'aws_link': f'https://lynnka-resources.s3.amazonaws.com/SensoryHistory_datasets/Benchmarks/Visualization/{vis_condition}s/{video_file}',
                                                   'Label': None,
                                                   'i_correct_choice': 'null'
                                                   })

    return pd.DataFrame(visualization_trial_definition)


def loadRSVPTrials(target_dir, datasets = ['RSVP_targetClips']):
    rsvp_trial_definition = []
    for dataset in datasets:
        df = pd.read_csv(f'{target_dir}/ExperimentalVideos/{dataset}-info.csv')
        for c in df['Count'].unique():
            _, video_number, video_id = df.loc[df['Count'] == c, 'Filename'].item()[:-4].split('_')
            for condition in ['RSVP_baseline', 'RSVP_full']:
                rsvp_trial_definition.append({'Video_ID': video_id,
                                                 'videoType': f'TargetClips-{condition}',
                                             'File_ID': f'TargetClips-{condition}_{c}_{video_number}_{video_id}',
                                                 'videoDuration (sec)': df.loc[df['Count'] == c, 'duration_clip_sec'].item(),
                                         #'startTime_clip (sec)': df.loc[df['Count'] == c, 'startTime_clip_sec'].item(),
                                         'Feedback_Training': 'null',
                                                 'Training': 'No',
                                                 'Video_file': None,
                                                 'aws_link': f'https://lynnka-resources.s3.amazonaws.com/SensoryHistory_datasets/{dataset}/{condition}/{c}_{video_number}_{video_id}.mp4',
                                                 'Label': None,
                                                 'i_correct_choice': 'null',
                                             #'concurrent_video_set_id':  df.loc[df['Count'] == c, 'concurrent_video_set_id'].item(),
                                             #'scenarios': df.loc[df['Count'] == c, 'scenarios'].item(),
                                             #'baseDuration': df.loc[df['Count'] == c, 'baseDuration'].item()
                                                 })
    return pd.DataFrame(rsvp_trial_definition)


def loadClipSeqTrials(target_dir, datasets = ['clipSequences'], conditions=['Baseline', 'Baseline_long', 'Seq_v0', 'Seq_v1', 'Seq_v2', 'Seq_v3', 'SeqRapid_v0', 'Seq_v1-replication']):
    seq_trial_definition = []
    for dataset in datasets:
        df = pd.read_csv(f'{target_dir}/ExperimentalVideos/{dataset}-info.csv')
        for c in df['Count'].unique():
            _, video_number, video_id = df.loc[(df['Version'] == 0) & (df['Count'] == c) & (df['video_type'] == 'Seq_v0'), 'Filename'].item()[:-4].split('_')
            for condition in conditions:
                if condition.endswith('-replication'):
                    condition_replication = condition.split('-')[0]
                    seq_trial_definition.append({'Video_ID': video_id,
                                                 'videoType': f'{dataset}-{condition}',
                                                 'File_ID': f'{dataset}-{condition}_{c}_{video_number}_{video_id}',
                                                 'videoDuration (sec)': df.loc[
                                                     (df['Version'] == 0) & (df['Count'] == c) & (df[
                                                                                                      'video_type'] == 'Seq_v0'), 'Target_clip_sec'].item() if c == 'Baseline' else
                                                 df.loc[(df['Version'] == 0) & (df['Count'] == c) & (
                                                             df['video_type'] == 'Seq_v0'), 'duration_clip_sec'].item(),
                                                 'Feedback_Training': 'null',
                                                 'Training': 'No',
                                                 'Video_file': None,
                                                 'aws_link': f'https://lynnka-resources.s3.amazonaws.com/SensoryHistory_datasets/{dataset}/{condition_replication}/{c}_{video_number}_{video_id}.mp4' if condition.startswith(
                                                     'Seq') else f'https://lynnka-resources.s3.amazonaws.com/SensoryHistory_datasets/{dataset}/{condition_replication}/{video_number}_{video_id}.mp4',
                                                 'Label': None,
                                                 'i_correct_choice': 'null',

                                                 })

                else:
                    seq_trial_definition.append({'Video_ID': video_id,
                                                     'videoType': f'{dataset}-{condition}',
                                                 'File_ID': f'{dataset}-{condition}_{c}_{video_number}_{video_id}',
                                                     'videoDuration (sec)': df.loc[(df['Version'] == 0) & (df['Count'] == c) & (df['video_type'] == 'Seq_v0'), 'Target_clip_sec'].item() if c == 'Baseline' else df.loc[(df['Version'] == 0) & (df['Count'] == c) & (df['video_type'] == 'Seq_v0'), 'duration_clip_sec'].item(),
                                                'Feedback_Training': 'null',
                                                     'Training': 'No',
                                                     'Video_file': None,
                                                     'aws_link': f'https://lynnka-resources.s3.amazonaws.com/SensoryHistory_datasets/{dataset}/{condition}/{c}_{video_number}_{video_id}.mp4' if condition.startswith('Seq') else f'https://lynnka-resources.s3.amazonaws.com/SensoryHistory_datasets/{dataset}/{condition}/{video_number}_{video_id}.mp4',
                                                     'Label': None,
                                                     'i_correct_choice': 'null',

                                                     })

    return pd.DataFrame(seq_trial_definition)



