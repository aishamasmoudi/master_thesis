from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from utils import compute_meanHR
from data_loaders import load_humanTrialData, load_humanReportRates, load_humanTrialData_frames
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
cm = 1 / 2.54

def compute_reliability(n, df, rep, metrics = ['correlation', 'MSE', 'r2'], grouping_variable=None, level='all',
                        type_variable='video_type', report_bins =None,
                        compute_categories=False):

    categories = df.loc[df.index[0], 'categories']
    reliability = []

    df_report_rates = df.groupby(['File_ID', type_variable], as_index=False)['final_choice'].apply(
        lambda x: compute_meanHR(x))
    df_split1 = df.groupby(['File_ID'], as_index=False).sample(n=n // 2, replace=False)
    df_split2 = df.loc[~df.index.isin(df_split1.index)].groupby(['File_ID'], as_index=False).sample(n=n // 2,
                                                                                                    replace=False)
    means_split1 = df_split1.groupby(['File_ID', type_variable], as_index=False)['final_choice'].apply(
        lambda x: compute_meanHR(x))
    means_split2 = df_split2.groupby(['File_ID',type_variable], as_index=False)['final_choice'].apply(
        lambda x: compute_meanHR(x))
    if grouping_variable is not None:
        means_split1[grouping_variable] = means_split1['File_ID'].map(dict(zip(df['File_ID'].to_list(), df[grouping_variable].to_list() )))
        groups = means_split1[grouping_variable].unique()
        groups = groups[pd.notna(groups)]

    means_split1.rename(columns={'final_choice': 'split1'}, inplace=True)
    means_split1['split2'] = means_split2['final_choice']
    means_split1['report_rate'] = df_report_rates['final_choice']
    means_split1['split2_shuffled'] = means_split2['final_choice'].sample(frac=1, replace=False).reset_index(drop=True)

    for metric in metrics:
        if level == 'video':

            means_split1[metric] = means_split1.apply(eval(f'safe_row_{metric}'), axis=1, feature1='split1',
                                                      feature2='split2')
            means_split1[f'{metric}_shuffled'] = means_split1.apply(eval(f'safe_row_{metric}'), axis=1, feature1='split1',
                                                                    feature2='split2_shuffled')

            reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                'Reliability': means_split1[metric].mean(),
                                'metric': metric,
                                'kind': 'all'})

            reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                'Reliability': means_split1[f'{metric}_shuffled'].mean(),
                                'metric': metric,
                                'kind': 'shuffled'})
        elif level == 'all':
            # bootstrap over file IDs
            if grouping_variable is not None:
                means_split1_resampled = means_split1.groupby([grouping_variable]).sample(frac=1, replace=True)
            else:
                means_split1_resampled = means_split1.sample(frac=1, replace=True)
            means_split1_resampled['Video_number'] = np.arange(len(means_split1_resampled))

            # convert it long format and compare all responses at once.
            if grouping_variable is not None:
                id_vars = ['Video_number', 'File_ID', type_variable, grouping_variable]
            else:
                id_vars = ['Video_number', 'File_ID', type_variable]
            means_long =  means_split1_resampled.melt(id_vars=id_vars,value_vars=['split1', 'split2', 'split2_shuffled', 'report_rate'],
                              var_name='Split', value_name='Responses')
            # Expand rates into separate columns
            means_long[categories] = means_long['Responses'].apply(pd.Series)
            # Again convert into even longer frame
            id_vars.append('Split')
            means_longer = means_long.melt(id_vars=id_vars, value_vars=categories,
                                                         var_name='Category', value_name='Response')

            id_vars = id_vars[:-1]
            id_vars.append('Category')
            # Finally put different splits into different value columns
            means_longer = means_longer.pivot(index=id_vars, columns='Split', values='Response')

            if report_bins is not None:
                means_longer['rate_bins'], report_bins = pd.cut(means_longer['report_rate'], include_lowest=True,
                                                                   bins=report_bins,
                                                                   retbins=True)

                min_count = means_longer.groupby('rate_bins')['report_rate'].count().min()
                means_longer = means_longer.groupby(
                    ['rate_bins'], observed=False).sample(min_count, replace=False)

                means_longer.drop(columns=['rate_bins'], inplace=True)

            if metric == 'correlation':
                reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                    'Reliability': means_longer.corr().loc['split1', 'split2'],
                                    'metric': metric,
                                    'kind': 'all',
                                    'category': 'all'})


                reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                    'Reliability': means_longer.corr().loc['split1', 'split2_shuffled'],
                                    'metric': metric,
                                    'kind': 'shuffled',
                                    'category': 'all'})
                if compute_categories:
                    for category in categories:
                        reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                            'Reliability': means_longer.xs(category, level='Category').corr().loc['split1', 'split2'],
                                            'metric': metric,
                                            'kind': 'all',
                                            'category': category})

            elif metric == 'MSE':

                reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                        'Reliability':mean_squared_error(means_longer['split1'].values, means_longer['split2'].values),
                                        'metric': metric,
                                        'kind': 'all',
                                    'category': 'all'})

                reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                        'Reliability': mean_squared_error(means_longer['split1'].values, means_longer['split2_shuffled'].values),
                                        'metric': metric,
                                        'kind': 'shuffled',
                                    'category': 'all'})

                if compute_categories:
                    for category in categories:
                        reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                            'Reliability': mean_squared_error(means_longer.xs(category, level='Category')['split1'].values,
                                                                    means_longer.xs(category, level='Category')['split2'].values),
                                            'metric': metric,
                                            'kind': 'all',
                                            'category': category})
            elif metric == 'r2':
                reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                    'Reliability': r2_score(means_longer['split1'].values,
                                                                              means_longer['split2'].values),
                                    'metric': metric,
                                    'kind': 'all',
                                    'category': 'all'})

                reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                    'Reliability': r2_score(means_longer['split1'].values,
                                                                              means_longer[
                                                                                  'split2_shuffled'].values),
                                    'metric': metric,
                                    'kind': 'shuffled',
                                    'category': 'all'})
                if compute_categories:
                    for category in categories:
                        reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                            'Reliability': r2_score(means_longer.xs(category, level='Category')['split1'].values,
                                                                    means_longer.xs(category, level='Category')['split2'].values),
                                            'metric': metric,
                                            'kind': 'all',
                                            'category': category})

        if grouping_variable != None:
            if level == 'video':
                group_means = means_split1.groupby([grouping_variable])[metric].mean()
                for group in means_split1[grouping_variable].unique():
                    reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                        'Reliability': group_means.loc[group],
                                        'metric': metric,
                                        'kind': group,
                                        'category': 'all'
                                        })
            elif level == 'all':

                for group in groups:

                    if metric == 'correlation':
                        reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                            'Reliability': means_longer.xs(group, level=grouping_variable).corr().loc[
                                                'split1', 'split2'],
                                            'metric': metric,
                                            'kind': group,
                                            'category': 'all'})

                        if compute_categories:
                            for category in categories:
                                reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                                    'Reliability': means_longer.xs(category, level='Category').xs(group,
                                                                                                                  level=grouping_variable).corr().loc[
                                                        'split1', 'split2'],
                                                    'metric': metric,
                                                    'kind': group,
                                                    'category': category})

                    elif metric == 'MSE':

                        reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                            'Reliability':
                                                mean_squared_error(
                                                    means_longer.xs(group, level=grouping_variable)['split1'].values,
                                                    means_longer.xs(group, level=grouping_variable)['split2'].values),
                                            'metric': metric,
                                            'kind': group,
                                            'category': 'all'})
                        if compute_categories:
                            for category in categories:
                                reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                                    'Reliability': mean_squared_error(
                                                        means_longer.xs(category, level='Category').xs(group,
                                                                                                       level=grouping_variable)[
                                                            'split1'].values,
                                                        means_longer.xs(category,
                                                                        level='Category').xs(
                                                            group, level=grouping_variable)[
                                                            'split2'].values),
                                                    'metric': metric,
                                                    'kind': group,
                                                    'category': category})

                    elif metric == 'r2':
                        reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                            'Reliability': r2_score(
                                                means_longer.xs(group, level=grouping_variable)['split1'].values,
                                                means_longer.xs(group, level=grouping_variable)['split2'].values),
                                            'metric': metric,
                                            'kind': group,
                                            'category': 'all'})
                        if compute_categories:
                            for category in categories:
                                reliability.append({'Repetition': rep, 'Sample size': n // 2,
                                                    'Reliability': r2_score(
                                                        means_longer.xs(category, level='Category').xs(group,
                                                                                                       level=grouping_variable)[
                                                            'split1'].values,
                                                        means_longer.xs(category,
                                                                        level='Category').xs(
                                                            group, level=grouping_variable)[
                                                            'split2'].values),
                                                    'metric': metric,
                                                    'kind': group,
                                                    'category': category})


    return reliability


def compute_reliability_scaling(df,
                         sample_sizes=[2, 4, 6, 12, 18, 24, 30, 36, 42, 48],
                         repetitions =100,
                        n_jobs=-1,
                                grouping_variable='video_group',
                                level='all',
                                type_variable='video_type',
                                report_bins =None):
    # Estimate the SNR as a function of the sample size
    reliability = []

    for n in sample_sizes:
        print(n)
        results = Parallel(n_jobs=n_jobs)(delayed(compute_reliability)(n, df, rep, grouping_variable=grouping_variable, level=level, type_variable=type_variable, report_bins=report_bins) for rep in range(repetitions))
        tmp = [pd.DataFrame(r) for r in results]
        reliability.append(pd.concat(tmp))

    return pd.concat(reliability)


def apply_reliability_prediction(reliability, kinds=None, predicted_sampleSizes = None,metrics = ['correlation','MSE',  'r2'],
                                 include_category=False):
    reliability['type'] = 'data'

    def func_correlation(x, a, b, c):
        # x-shifted log
        return 1 - a * np.exp(-b * np.log(x + c))

    def func_r2(x, a, b, c):  # x-shifted log
        return 1 - a * np.exp(-b * np.log(x + c))

    def func_MSE(x, a, b, c):  # x-shifted log
        return a * np.exp(-b * np.log(x + c))

    if kinds is None:
        kinds = reliability['kind'].unique()
    if predicted_sampleSizes is None:
        predicted_sampleSizes = np.linspace(0, 80, 21)[1:]
    reliability_df = reliability.copy()
    for metric in metrics:
        for kind in kinds:
            if (kind == 'shuffled') | (include_category == False):
                for r in reliability['Repetition'].unique():
                    mean_vals = \
                    reliability.loc[(reliability['kind'] == kind) & (reliability['category'] == 'all')&(reliability['metric'] == metric) & (reliability['Repetition'] == r)].groupby('Sample size',
                                                                                                               as_index=False)[
                        'Reliability'].mean()
                    try:
                        xmin = mean_vals['Sample size'].values.min()
                        eps = 1e-6
                        lower_c = -xmin + eps  # e.g., for xmin=1.0 -> -0.999999
                        bounds_lower = (0.0, 0.0, lower_c)
                        bounds_upper = (10.0, 10.0, 10.0)  # wide but finite
                        p0=[0.5, 0.5, 0.1]

                        popt, pcov = curve_fit(eval(f'func_{metric}'), mean_vals['Sample size'].values,
                                               mean_vals['Reliability'].values,
                                               p0=p0,
                                               bounds=(bounds_lower, bounds_upper),
                                               maxfev=2000, )
                        prediction_df = pd.DataFrame({'Sample size': predicted_sampleSizes,
                                                      'kind': kind,
                                                      'Repetition': r,
                                                      'Reliability': eval(f'func_{metric}')(predicted_sampleSizes, *popt),
                                                      'metric': metric,
                                                      'type': 'prediction',
                                                      'category': 'all'
                                                      })

                        reliability_df = pd.concat([reliability_df, prediction_df], axis=0, ignore_index=True)
                    except Exception as e:
                        print(f"Error for {r} - {kind} - {metric}")
                        continue  # Optional - explicitly continue to next iteration
            else:
                for cat in reliability['category'].unique():
                    for r in reliability['Repetition'].unique():
                        mean_vals = \
                            reliability.loc[(reliability['category'] == cat) & (reliability['kind'] == kind) & (reliability['metric'] == metric) & (
                                        reliability['Repetition'] == r)].groupby('Sample size',
                                                                                 as_index=False)[
                                'Reliability'].mean()
                        try:
                            xmin = mean_vals['Sample size'].values.min()
                            eps = 1e-6
                            lower_c = -xmin + eps  # e.g., for xmin=1.0 -> -0.999999
                            bounds_lower = (0.0, 0.0, lower_c)
                            bounds_upper = (10.0, 10.0, 10.0)  # wide but finite
                            p0 = [0.5, 0.5, 0.1]

                            popt, pcov = curve_fit(eval(f'func_{metric}'), mean_vals['Sample size'].values,
                                                   mean_vals['Reliability'].values,
                                                   p0=p0,
                                                   bounds=(bounds_lower, bounds_upper),
                                                   maxfev=2000, )
                            prediction_df = pd.DataFrame({'Sample size': predicted_sampleSizes,
                                                          'kind': kind,
                                                          'Repetition': r,
                                                          'Reliability': eval(f'func_{metric}')(predicted_sampleSizes, *popt),
                                                          'metric': metric,
                                                          'type': 'prediction',
                                                          'category': cat
                                                          })

                            reliability_df = pd.concat([reliability_df, prediction_df], axis=0, ignore_index=True)
                        except Exception as e:
                            print(f"Error for {r} - {cat} - {kind} - {metric}")
                            continue  # Optional - explicitly continue to next iteration

    return reliability_df

def load_reliability_durations(N = 40, repetitions=1000, recompute=False, kinds=None, level='all',
                     result_dir='/braintree/home/lynnka/Projects/ContinuousRecognition_modeling/Results/',
                     scaling=True, duration_bins = [0.1, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 15]):
    filename = Path(f'{result_dir}Reliability_{level}_duration.csv')

    if (recompute == False) & (filename.exists()):
        reliability = pd.read_csv(filename)

    else:
        df = load_humanTrialData()
        trial_counts = df[~df['video_type'].isin(['Catch', 'practice'])].groupby('File_ID').size()

        df_selected = df.loc[
            (~df['video_type'].isin(['Catch', 'practice'])) & df['File_ID'].isin(
                trial_counts[trial_counts > N].index)]

        df_selected['videoDuration_bins'], bins = pd.cut(df_selected['videoDuration (sec)'],
                                                         bins=duration_bins, retbins=True)

        df_selected = df_selected.loc[df_selected['video_group']=='test']
        reliability = compute_reliability_scaling(df_selected, sample_sizes=np.arange(2, N + 1, 3),#n_jobs=1,
                                                  repetitions=repetitions, level='all', grouping_variable='videoDuration_bins')

        reliability.to_csv(filename, index=False)

    if  ('type' not in reliability.columns) & scaling:
        reliability = apply_reliability_prediction(reliability, include_category=False)

        reliability.to_csv(filename, index=False)

    return reliability

