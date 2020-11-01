import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

pd.set_option("display.max_rows", None, "display.max_columns", None)

train_file_path = './home-data-for-ml-course/train.csv'
train_data = pd.read_csv(train_file_path)
colsOriginal = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', '1stFlrSF', 'FullBath', 'BedroomAbvGr',
            'KitchenAbvGr', 'GarageArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2']
colsToTransfer = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
                  'Neighborhood', 'BldgType']
# , 'HouseStyle']
                #   'RoofMatl', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                # ]
features = colsOriginal + colsToTransfer
train_X = train_data[features]
for key in colsToTransfer:
    default_key = 'NoSeWa' if key == 'Utilities' else train_X[key].mode().iloc[0]
    train_X[key] = train_X[key].fillna(default_key)

train_X = train_X.fillna(train_X.mean())
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), colsToTransfer)], remainder='passthrough')
train_X = columnTransformer.fit_transform(train_X)
train_y = train_data.SalePrice

s_train_X, s_val_X, s_train_y, s_val_y = train_test_split(train_X, train_y, random_state=0)
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
# for key in colsToTransfer:
#     default_key = 'NoSeWa' if key == 'Utilities' else test_X[key].mode().iloc[0]
#     default_key = '2.5Fin' if key == 'HouseStyle' else default_key
#     test_X[key] = test_X[key].fillna(default_key)
# test_X = test_X.fillna(test_X.mean())
# test_X = columnTransformer.fit_transform(test_X)
# test_preds = rf_model.predict(test_X)
# output = pd.DataFrame({'Id': test_data.Id,
#                        'SalePrice': test_preds})
# output.to_csv('submission.csv', index=False)