import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.max_rows", None, "display.max_columns", None)

train_file_path = './home-data-for-ml-course/train.csv'
train_data = pd.read_csv(train_file_path)
features = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', '1stFlrSF', 'FullBath', 'BedroomAbvGr',
            'KitchenAbvGr', 'GarageArea']

train_X = train_data[features]
train_X['MSZoning'].loc[:] = LabelEncoder().fit_transform(train_X['MSZoning'].loc[:].values)
train_X = train_X.fillna(train_X.mean())
train_y = train_data.SalePrice

s_train_X, s_val_X, s_train_y, s_val_y = train_test_split(train_X, train_y, random_state=0)
print('Train count: %i %i' % (len(s_train_X), len(s_train_y)))
print('Test count: %i %i' % (len(s_val_X), len(s_val_y)))
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(s_train_X, s_train_y)

# rf_model.fit(train_X, train_y)
predicts = rf_model.predict(s_val_X)
mae = mean_absolute_error(predicts, s_val_y)
print('MAE %i' % mae)


# test_file_path = './home-data-for-ml-course/test.csv'
# test_data = pd.read_csv(test_file_path)
#
# test_X = test_data[features]
# test_X['MSZoning'] = test_X['MSZoning'].fillna('X')
# test_X['MSZoning'].loc[:] = LabelEncoder().fit_transform(test_X['MSZoning'].loc[:].values)
#
# test_X = test_X.fillna(train_X.mean())
# test_preds = rf_model.predict(test_X)
# output = pd.DataFrame({'Id': test_data.Id,
#                        'SalePrice': test_preds})
# output.to_csv('submission.csv', index=False)
