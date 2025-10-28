import numpy as np
import pandas as pd





df = pd.read_csv('dataset/churn_features_export.csv')

#print(df[['Frequency', 'Monetary', 'CustomerLifetimeDays']].describe())


# =============================================================================
#          Frequency       Monetary  CustomerLifetimeDays
# count  4338.000000    4338.000000           4338.000000
# mean      4.272015    2054.266459            221.831259
# std       7.697998    8989.230441            117.854135
# min       1.000000       3.750000              0.000000
# 25%       1.000000     307.415000            111.000000
# 50%       2.000000     674.485000            247.000000
# 75%       5.000000    1661.740000            325.000000
# max     209.000000  280206.020000            372.000000
# =============================================================================

n = len(df)



def quick_quantile_sample(df, n):
    df = df.dropna().values
    qdf = np.linspace(0, 1, n)  #n parçaya böldük
    qvals = np.quantile(df, qdf)    #yüzdelik dilim
    U = np.random.rand(n)   #random
    out = np.interp(U, qdf, qvals)      #interpolasyon
    return out

synthetic_df = pd.DataFrame({
    'CustomerID': np.arange(10000, 10000 + n),
    'Frequency': np.rint(quick_quantile_sample(df['Frequency'], n)).astype(int),
    'Monetary': np.round(quick_quantile_sample(df['Monetary'], n), 2),
    'CustomerLifetimeDays': np.rint(quick_quantile_sample(df['CustomerLifetimeDays'], n)).astype(int)
})


print(synthetic_df.describe())


# =============================================================================
#          CustomerID    Frequency       Monetary  CustomerLifetimeDays
# count   1000.000000  1000.000000    1000.000000            1000.00000
# mean   10499.500000     4.310000    2104.156080             228.34300
# std      288.819436     8.142125    9066.064046             116.16246
# min    10000.000000     1.000000      16.020000               0.00000
# 25%    10249.750000     1.000000     305.392500             126.75000
# 50%    10499.500000     2.000000     613.125000             254.00000
# 75%    10749.250000     5.000000    1660.910000             332.00000
# max    10999.000000   183.000000  206251.600000             372.00000
# =============================================================================


# =============================================================================
# df.loc[df['Frequency'] >= 10, 'Monetary'].min() #10dan büyük olanlar 1500 olmalı
# df.loc[df['Frequency'] >= 15, 'Monetary'].min()#2000
# df.loc[df['Frequency'] >= 30, 'Monetary'].min()#4000
# df.loc[df['Frequency'] >= 50, 'Monetary'].min() #8500
# df.loc[df['Frequency'] >= 70, 'Monetary'].min() #11k
# df.loc[df['Frequency'] >= 90, 'Monetary'].min()#12k
# =============================================================================

filtered_df = synthetic_df.copy()

filtered_df = filtered_df[
    ~(
        ((filtered_df['Frequency'] > 10) & (filtered_df['Monetary'] < 1500)) |
        ((filtered_df['Frequency'] > 15) & (filtered_df['Monetary'] < 2000)) |
        ((filtered_df['Frequency'] > 30) & (filtered_df['Monetary'] < 4000)) |
        ((filtered_df['Frequency'] > 70) & (filtered_df['Monetary'] < 11000)) |
        ((filtered_df['Frequency'] > 90) & (filtered_df['Monetary'] < 12000)) |
        ((filtered_df['Frequency'] > filtered_df['CustomerLifetimeDays']))   
    )
]




filtered_df.to_csv("filtered_dataset.csv", index=False)















