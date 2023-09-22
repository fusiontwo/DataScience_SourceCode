# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:05:27 2023

@author: yereum
"""
# 미완성본임!!!

import pandas as pd
import numpy as np

def adjust_data_frame(df):
    df['Overall rank'] = df['Overall rank'].astype('int')
    df['Country'] = df['Country']
    df['Score'] = df['Score'].astype('float')
    df['GDP per capita'] = df['GDP per capita'].astype('float')
    df['Social support'] = df['Social support'].astype('float')
    df['Healthy life expectancy'] = df['Healthy life expectancy'].astype('float')
    df['Freedom to make life choices'] = df['Freedom to make life choices'].astype('float')
    df['Generosity'] = df['Generosity'].astype('float')
    df['Perceptions of corruption'] = df['Perceptions of corruption'].astype('float')


def mean_median_mode(df, attr_list):
    mean = {}
    median = {}
    mode = {}
    
    for c in ['Males', 'Females', 'Birth rate', 'Death rate']:
        mean[c] = df[c].mean()
        median[c] = df[c].median()
        mode[c] = dict(df[c].mode())
        
    print(f"mean = {mean}\n")
    print(f"median = {median}\n")
    print(f"mode = {mode}\n")
    
def std_var(df):
    std = {}
    var = {}

    
    for c in ['Males', 'Females', 'Birth rate', 'Death rate']:
        std[c] = df[c].std()
        var[c] = df[c].var()

        
    print(f"std = {std}\n")
    print(f"var = {var}\n")



import matplotlib.pyplot as plt 

def percentile(df):

    p = [x for x in range(0, 101, 10)]

    for c in ['Males', 'Females', 'Birth rate', 'Death rate']:
        percentile = np.percentile(df[c], p)
        plt.plot(p, percentile, 'o-')
        plt.xlabel('percentile')
        plt.ylabel(c)
        plt.xticks(p)
        plt.yticks(np.arange(0, max(percentile)+1, max(percentile)/10.0))
        plt.show()


def boxplot(df):
    boxplot = df[['Males', 'Females', 'Birth rate', 'Death rate']].boxplot()
    plt.show()  

def histogram(df):
    for c in ['Males', 'Females', 'Birth rate', 'Death rate']:
        plt.hist(df[c], facecolor='blue', bins=10)  # bins : 막대 개수
        plt.xlabel(c)
        plt.show()
        
def scatter_plot(df, attr_list1, attr_list2):
    for c1 in ['Males', 'Females', 'Birth rate', 'Death rate']: 
        for c2 in ['Males', 'Females', 'Birth rate', 'Death rate']:
            if c1 == c2:
                continue
            plt.scatter(df[c1], df[c2])
            plt.xlabel(c1)
            plt.ylabel(c2)
            plt.show()


def pairplot(df):
    import seaborn as sns
    sns.pairplot(df[['Males', 'Females', 'Birth rate', 'Death rate']])
    
    
if __name__ == '__main__':
    csv_file = '2019 happiness index.csv'
    
    df = pd.read_csv(csv_file)
    adjust_data_frame(df)
    
    df = pd.merge(df1, df2, on='Country', how='inner')
    attr_list = attr_list1 + attr_list2
    
    df.to_csv(merged.csv)
    
    #mean_median_mode(df)
    #std_var(df)  # 각 나라의 Birth rate은 편차가 심하고, Death rate은 편차가 적다.
    #percentile(df)
    #boxplot(df)
    #histogram(df)
    #scatter_plot(df)
    #pairplot(df)
    
    
    