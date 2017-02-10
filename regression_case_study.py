import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('data/Train.csv')
'''
for column in df.columns:
    uniques = df[column].unique()
    if len(uniques) < 100:
        prices = []
        for unique in uniques:
            prices = df[df[column]==unique]['SalePrice']
            print column, ":", unique, "-- Mean:", np.mean(prices), "Std:", np.std(prices)
            #df[column + ":" + unique] = df[column]==unique
'''
def clean_data(df):
    df=df.loc[df['YearMade']>1930]
    y = df['SalePrice']
    df = df[['auctioneerID', 'YearMade', 'MachineHoursCurrentMeter', 'state', 'ProductGroup', 'ProductSize', 'fiProductClassDesc', 'SalePrice']]

def categorize_column(df, column):
    for unique in df[column].unique():
        df[unique] = (df[column]==unique).astype(int)
    return df
'''
categorize_column(df, 'state')
print df.head()

def predict_values(df, predictor, predicted):
    counts = df.groupby(predictor).count(predicted)
    df[pd.isnull(df[predicted])][predicted] =

counts = df.groupby(['fiProductClassDesc', 'ProductSize']).count()
counts.reset_index()
for value in df['fiProductClassDesc'].unique():

counts[counts['fiProductClassDesc'] == value]

df['Skid Steer Loader - 0.0 to 701.0 Lb Operating Capacity', ]

a = df[df['fiProductClassDesc'].isin(['Track Type Tractor, Dozer - 105.0 to 130.0 Horsepower',
                                       'Track Type Tractor, Dozer - 130.0 to 160.0 Horsepower',
                                       'Track Type Tractor, Dozer - 160.0 to 190.0 Horsepower',
                                       'Track Type Tractor, Dozer - 190.0 to 260.0 Horsepower',
                                       'Track Type Tractor, Dozer - 20.0 to 75.0 Horsepower',
                                       'Track Type Tractor, Dozer - 260.0 + Horsepower',
                                       'Track Type Tractor, Dozer - 75.0 to 85.0 Horsepower',
                                       'Track Type Tractor, Dozer - 85.0 to 105.0 Horsepower'])]
df.groupby('ProductSize').mean()

c = df[df['fiProductClassDesc'].isin(['Hydraulic Excavator, Track - 0.0 to 2.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 11.0 to 12.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 12.0 to 14.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 14.0 to 16.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 150.0 to 300.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 16.0 to 19.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 19.0 to 21.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 2.0 to 3.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 21.0 to 24.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 24.0 to 28.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 28.0 to 33.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 3.0 to 4.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 300.0 + Metric Tons',
                                       'Hydraulic Excavator, Track - 33.0 to 40.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 4.0 to 5.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 4.0 to 6.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 40.0 to 50.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 5.0 to 6.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 50.0 to 66.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 6.0 to 8.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 66.0 to 90.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 8.0 to 11.0 Metric Tons',
                                       'Hydraulic Excavator, Track - 90.0 to 150.0 Metric Tons',
                                       'Hydraulic Excavator, Track - Unidentified',
                                       'Hydraulic Excavator, Track - Unidentified (Compact Construction)'])]
'''

dictionary = {'fiProductClassDesc': {'Backhoe Loader - 0.0 to 14.0 Ft Standard Digging Depth': 1,
                               'Backhoe Loader - 14.0 to 15.0 Ft Standard Digging Depth': 2,
                               'Backhoe Loader - 15.0 to 16.0 Ft Standard Digging Depth': 3,
                               'Backhoe Loader - 16.0 + Ft Standard Digging Depth': 4,
                               'Backhoe Loader - Unidentified': 2.5,
                               'Hydraulic Excavator, Track - 0.0 to 2.0 Metric Tons': 1,
                               'Hydraulic Excavator, Track - 11.0 to 12.0 Metric Tons': 8,
                               'Hydraulic Excavator, Track - 12.0 to 14.0 Metric Tons': 9,
                               'Hydraulic Excavator, Track - 14.0 to 16.0 Metric Tons': 10,
                               'Hydraulic Excavator, Track - 150.0 to 300.0 Metric Tons': 21,
                               'Hydraulic Excavator, Track - 16.0 to 19.0 Metric Tons': 11,
                               'Hydraulic Excavator, Track - 19.0 to 21.0 Metric Tons': 12,
                               'Hydraulic Excavator, Track - 2.0 to 3.0 Metric Tons': 2,
                               'Hydraulic Excavator, Track - 21.0 to 24.0 Metric Tons': 13,
                               'Hydraulic Excavator, Track - 24.0 to 28.0 Metric Tons': 14,
                               'Hydraulic Excavator, Track - 28.0 to 33.0 Metric Tons': 15,
                               'Hydraulic Excavator, Track - 3.0 to 4.0 Metric Tons': 3,
                               'Hydraulic Excavator, Track - 300.0 + Metric Tons': 22,
                               'Hydraulic Excavator, Track - 33.0 to 40.0 Metric Tons': 16,
                               'Hydraulic Excavator, Track - 4.0 to 5.0 Metric Tons': 4,
                               'Hydraulic Excavator, Track - 4.0 to 6.0 Metric Tons': 4,
                               'Hydraulic Excavator, Track - 40.0 to 50.0 Metric Tons': 17,
                               'Hydraulic Excavator, Track - 5.0 to 6.0 Metric Tons': 5,
                               'Hydraulic Excavator, Track - 50.0 to 66.0 Metric Tons': 18,
                               'Hydraulic Excavator, Track - 6.0 to 8.0 Metric Tons': 6,
                               'Hydraulic Excavator, Track - 66.0 to 90.0 Metric Tons': 19,
                               'Hydraulic Excavator, Track - 8.0 to 11.0 Metric Tons': 7,
                               'Hydraulic Excavator, Track - 90.0 to 150.0 Metric Tons': 20,
                               'Hydraulic Excavator, Track - Unidentified': 10,
                               'Hydraulic Excavator, Track - Unidentified (Compact Construction)': 10,
                               'Motorgrader - 130.0 to 145.0 Horsepower': 2,
                               'Motorgrader - 145.0 to 170.0 Horsepower': 3,
                               'Motorgrader - 170.0 to 200.0 Horsepower': 4,
                               'Motorgrader - 200.0 + Horsepower': 5,
                               'Motorgrader - 45.0 to 130.0 Horsepower': 1,
                               'Motorgrader - Unidentified': 3,
                               'Skid Steer Loader - 0.0 to 701.0 Lb Operating Capacity': 1,
                               'Skid Steer Loader - 1251.0 to 1351.0 Lb Operating Capacity': 4,
                               'Skid Steer Loader - 1351.0 to 1601.0 Lb Operating Capacity': 5,
                               'Skid Steer Loader - 1601.0 to 1751.0 Lb Operating Capacity': 6,
                               'Skid Steer Loader - 1751.0 to 2201.0 Lb Operating Capacity': 7,
                               'Skid Steer Loader - 2201.0 to 2701.0 Lb Operating Capacity': 8,
                               'Skid Steer Loader - 2701.0+ Lb Operating Capacity': 9,
                               'Skid Steer Loader - 701.0 to 976.0 Lb Operating Capacity': 2,
                               'Skid Steer Loader - 976.0 to 1251.0 Lb Operating Capacity': 3,
                               'Skid Steer Loader - Unidentified': 5,
                               'Track Type Tractor, Dozer - 105.0 to 130.0 Horsepower': 4,
                               'Track Type Tractor, Dozer - 130.0 to 160.0 Horsepower': 5,
                               'Track Type Tractor, Dozer - 160.0 to 190.0 Horsepower': 6,
                               'Track Type Tractor, Dozer - 190.0 to 260.0 Horsepower': 7,
                               'Track Type Tractor, Dozer - 20.0 to 75.0 Horsepower': 1,
                               'Track Type Tractor, Dozer - 260.0 + Horsepower': 8,
                               'Track Type Tractor, Dozer - 75.0 to 85.0 Horsepower': 2,
                               'Track Type Tractor, Dozer - 85.0 to 105.0 Horsepower': 3,
                               'Track Type Tractor, Dozer - Unidentified': 4.5,
                               'Wheel Loader - 0.0 to 40.0 Horsepower': 1,
                               'Wheel Loader - 100.0 to 110.0 Horsepower': 6,
                               'Wheel Loader - 1000.0 + Horsepower': 18,
                               'Wheel Loader - 110.0 to 120.0 Horsepower': 7,
                               'Wheel Loader - 120.0 to 135.0 Horsepower': 8,
                               'Wheel Loader - 135.0 to 150.0 Horsepower': 9,
                               'Wheel Loader - 150.0 to 175.0 Horsepower': 10,
                               'Wheel Loader - 175.0 to 200.0 Horsepower': 11,
                               'Wheel Loader - 200.0 to 225.0 Horsepower': 12,
                               'Wheel Loader - 225.0 to 250.0 Horsepower': 13,
                               'Wheel Loader - 250.0 to 275.0 Horsepower': 14,
                               'Wheel Loader - 275.0 to 350.0 Horsepower': 15,
                               'Wheel Loader - 350.0 to 500.0 Horsepower': 16,
                               'Wheel Loader - 40.0 to 60.0 Horsepower': 2,
                               'Wheel Loader - 500.0 to 1000.0 Horsepower': 17,
                               'Wheel Loader - 60.0 to 80.0 Horsepower': 3,
                               'Wheel Loader - 80.0 to 90.0 Horsepower': 4,
                               'Wheel Loader - 90.0 to 100.0 Horsepower': 5,
                               'Wheel Loader - Unidentified': 10} }
df.replace(to_replace=dictionary, inplace=True)
df = df[['ProductGroup', 'SalesID']]
df.to_csv('productsize_dict.csv')
