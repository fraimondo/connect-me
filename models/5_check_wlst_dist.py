from utils import get_data
import pandas as pd
df_all = get_data(fmri=True, eeg_visual=True, eeg_abcd=True,
                  eeg_model=True, eeg_features=True)
df = pd.read_csv('../data/dataset.csv', sep=';', index_col='Id')
df_all.join(df['WLST'].dropna())
check = df_all.join(df['WLST'].dropna())

to_keep = ['doc.enrollment', 'doc.discharge', 'WLST']
check = check[to_keep]

print('Enrollment')
print(
    check[['doc.enrollment', 'WLST']].reset_index().groupby(
        ['doc.enrollment', 'WLST']).count()
)
print('Discharge')
print(
    check[['doc.discharge', 'WLST']].reset_index().groupby(
        ['doc.discharge', 'WLST']).count()
)
