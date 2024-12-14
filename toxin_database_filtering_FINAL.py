# November 13, 2023 -- Mariam Gelashvili
# Toxin Database Filtering

#!/usr/bin/env python3

import pandas as pd
import regex as re
import sys


def loadup(filename):
    tox_df = pd.read_csv(filename)

    return tox_df


def nonstanAAcheck(df):
    nonstanAAlist = ['B', 'X', 'Z', 'J']
    AAlist = '|'.join(nonstanAAlist)

    df['nonstandardAADetected'] = df['sequence'].str.contains(AAlist)
    df.drop(df[df['nonstandardAADetected'] == True].index, inplace=True)
    # dropped sequences are not retained
    return df


def rangeProblem(df):
    df['unknownSignal'] = df['signalPeptide'].str.contains('\?') |\
                          df['signalPeptide'].str.contains('<', na=False) |\
                          df['peptide'].str.contains('>', na=False)

    df.drop(df[df['unknownSignal'] == True].index, inplace=True)
    # dropped sequences are not retained
    print(df)

    return df


def signalRangeExtract(df):
    df['signalRange'] = df['signalPeptide'].str.findall('(\d+)')
    df1 = df.dropna(subset=['signalRange'])
    df1['signalStart'] = df1['signalRange'].str[0].astype(int)
    df1['signalEnd'] = df1['signalRange'].str[1].astype(int)
    df1['signal'] = df1.apply(lambda x: x['sequence'][x['signalStart']-1:x['signalEnd']], 1)

    df.drop('signalRange', axis=1, inplace=True)
    df1.drop('signalRange', axis=1, inplace=True)

    result = pd.merge(df, df1, how="left")

    return result


def proRangeExtract(df):
    df['proRange'] = df['proPeptide'].str.findall('(\d+)')
    df1 = df.dropna(subset=['proRange'])
    df1['proStart'] = df1['proRange'].str[0].astype(int)
    df1['proEnd'] = df1['proRange'].str[1].astype(int)
    df1['pro'] = df1.apply(lambda x: x['sequence'][x['proStart']-1:x['proEnd']], 1)

    df.drop('proRange', axis=1, inplace=True)
    df1.drop('proRange', axis=1, inplace=True)

    result = pd.merge(df, df1, how="left")

    return result


def peptideExtract(df):
    df['pepRange'] = df['peptide'].str.findall('(\d+)')
    df1 = df.dropna(subset=['pepRange'])
    df1['pepStart'] = df1['pepRange'].str[0].astype(int)
    df1['pepEnd'] = df1['pepRange'].str[1].astype(int)
    df1['pep'] = df1.apply(lambda x: x['sequence'][x['pepStart']-1:x['pepEnd']], 1)

    df.drop('pepRange', axis=1, inplace=True)
    df1.drop('pepRange', axis=1, inplace=True)

    result = pd.merge(df, df1, how="left")

    return result


def autoMature(df):
    testd = df[~(df.dropna(subset=['signalPeptide', 'proPeptide', 'peptide'], how='all'))]
    
    return testd
# if all ranges arent there, assume its the mature



def main():
    tox_df1 = loadup(sys.argv[1])
    tox_df2 = rangeProblem(tox_df1)
    tox_df3 = signalRangeExtract(tox_df2)
    tox_df4 = proRangeExtract(tox_df3)
    tox_df5 = peptideExtract(tox_df4)
    
    tox_df5.to_csv('ConeTox_filtered.tsv', sep='\t')


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
