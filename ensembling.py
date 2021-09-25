import pandas as pd
import numpy as np

cat = pd.read_csv('C:/Users/Radmir/Desktop/raifhack/submission_cat_ohe.csv')
lgb = pd.read_csv('C:/Users/Radmir/Desktop/raifhack/submission_lgb_ohe.csv')
xgb = pd.read_csv('C:/Users/Radmir/Desktop/raifhack/submission_xgb_ohe.csv')

sub = lgb.copy()
cat['per_square_meter_price'] = cat['per_square_meter_price'].values*((3*np.pi/10 + 0.0001*np.e)/1.1)
xgb['per_square_meter_price'] = xgb['per_square_meter_price'].values*((3*np.pi/10 + 0.0001*np.e)/1.1)
lgb['per_square_meter_price'] = lgb['per_square_meter_price'].values*(3*np.pi/10 + 0.0001*np.e)
sub['per_square_meter_price'] = cat['per_square_meter_price'].values*0.05 + xgb['per_square_meter_price']*0.15 + \
    lgb['per_square_meter_price'].values*0.8
sub.to_csv('C:/Users/Radmir/Desktop/raifhack/submission_ensemble_test_post_processing_ohe.csv', index=False)