import datetime
import os
import re

import pandas as pd

from global_vars import *


def datetime_handler(cell: str) -> float | None:
    cell = cell.replace(',', '.')

    if re.fullmatch(r'\b\d{4}-\d\d-\d\d\b', cell):
        pass

    elif re.fullmatch(r'\b[а-я]{3}\.\d\d\b', cell):
        cell = cell.split('.')
        m = MONTHS[cell[0]]
        y = cell[1]
        if int(y) > int(TODAY_YEAR[-2:]):
            y = '19' + y
        else:
            y = '20' + y
        cell = f'{y}-{m}-01'

    elif re.fullmatch(r'\b\d\d\.\d\d\.\d{4}\b', cell):
        cell = cell.split('.')
        cell = f'{cell[2]}-{cell[1]}-{cell[0]}'

    elif re.fullmatch(r'\b\d\d\.\d{4}\b', cell):
        cell = cell.split('.')
        cell = f'{cell[1]}-{cell[0]}-01'

    elif re.fullmatch(r'\b[а-я]{3,8} \d{4}\b', cell):
        cell = cell.split(' ')
        m = MONTHS[cell[0]]
        y = cell[1]
        cell = f'{y}-{m}-01'

    elif re.fullmatch(r'\b\d\d\d\d г\.', cell):
        cell = f'{cell.split(" ")[0]}-01-01'

    elif re.fullmatch(r'\b\d{4}\b', cell):
        cell = f'{cell}-01-01'

    else:
        cell = None

    if cell:
        cell = datetime.strptime(cell, "%Y-%m-%d").date()

    return cell


def predprocessing(
             path_participants: str,
             path_metrics: str,
             date_column_names: list = False
             ):
    """
    The class allows to create an object containing all participants & metrics for further clusterization and analysis

    :param path_participants: path to csv-file containing information on participants
    :param path_metrics: path to directory containing csv-files containing information on metrics
    :param date_column_names: names of columns of date-type
    """

    df_participants = pd.read_csv(path_participants)
    dfs_metrics = []

    df_participants.fillna('', inplace=True)

    if date_column_names:
        for date_column_name in date_column_names:
            new_values = [datetime_handler(n) for n in df_participants[date_column_name].tolist()]
            df_participants[date_column_name] = pd.DataFrame(new_values, columns=[date_column_name])

    filenames_metrics = os.listdir(path_metrics)
    filenames_metrics.sort()

    for filename_metrics in filenames_metrics:
        df_metrics = pd.read_csv(f'{path_metrics}/{filename_metrics}')
        ss = df_metrics.columns.tolist()

        if 'Баллы, %' not in ss:
            print(f'No "Баллы, %" in "{filename_metrics}"')
            continue
        if 'ФИО участников' not in ss and 'ФИО' not in ss:
            print(f'No "ФИО" or "ФИО участников"  in "{filename_metrics}"')
            continue
        if df_metrics['Баллы, %'].isna().any():
            print(f'No values in "Баллы, %" in "{filename_metrics}"')
            continue

        if 'ФИО участников' in ss:
            df_metrics['ФИО участников'] = df_metrics['ФИО участников'].str.split('; ')
            df_metrics = df_metrics.explode('ФИО участников')
            df_metrics = df_metrics.rename(columns={'ФИО участников': 'ФИО', })

        dfs_metrics.append(df_metrics)

        df_metrics['Баллы, %'] = df_metrics['Баллы, %'].str.replace(',', '.')
        df_metrics['Баллы, %'] = df_metrics['Баллы, %'].str.replace('%', '')
        df_metrics['Баллы, %'] = df_metrics['Баллы, %'].astype('float64')

        df_metrics = df_metrics[['ФИО', 'Баллы, %']]
        df_metrics = df_metrics.rename(columns={'Баллы, %': filename_metrics.replace('.csv', '')})
        df_participants = df_participants.merge(right=df_metrics, on='ФИО', how='left')

    df_participants.to_excel('result.xlsx')


predprocessing(
    path_participants='Data/Участники anonimized.csv',
    path_metrics='Data/Anonimized',
    date_column_names=[
        'Начало трудового стажа',
        'Дата рождения',
        'Начало трудовой деятельности в РОСАТОМ'
    ]
)
