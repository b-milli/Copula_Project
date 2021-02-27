import pandas as pd
import itertools

df = pd.read_csv('final_qtrly_joined.csv')
keys = pd.read_excel('keys.xlsx')

a = [keys['CREDIT_BUCKET'].dropna(),keys['cur_qtr'].dropna(),keys['newloan'].dropna()]
keys = list(itertools.product(*a))

buckets = pd.DataFrame(keys, columns =['CREDIT_BUCKET', 'QTR', 'NEW_LOAN'])
buckets['PD'] = buckets.apply(lambda _: '', axis=1)

def data_filter(bucket,qtr,nl):
    filtered_df = df[(df['CREDIT_BUCKET']==bucket) & (df['cur_qtr']==qtr) & (df['new_loan']==nl)]
    delinq_count = len(filtered_df[df['DELINQ_IND']==1].index)
    total_count = max(1,len(filtered_df.index))
    return delinq_count/total_count

for i in range(buckets.shape[0]):
    buckets['PD'].iloc[i] = data_filter(buckets.iloc[i][0],buckets.iloc[i][1],buckets.iloc[i][2])
    
buckets.to_excel('buckets.xlsx')
