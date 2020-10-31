import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

pd.set_option("display.max_rows", None, "display.max_columns", None)

train_file_path = './home-data-for-ml-course/train.csv'
train_data = pd.read_csv(train_file_path)
train_data = pd.get_dummies(train_data)

train_features = ['MSSubClass', 'LotFrontage', 'LotArea', 'YearBuilt', '1stFlrSF', 'FullBath', 'BedroomAbvGr',
            'KitchenAbvGr', 'GarageArea']
train_X = train_data[train_features]
train_X = np.nan_to_num(train_X)
train_y = np.nan_to_num(train_data.SalePrice)
# train_y = train_data.SalePrice
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)

test_file_path = './home-data-for-ml-course/test.csv'
test_data = pd.read_csv(test_file_path)
test_data = pd.get_dummies(test_data)
test_features = ['MSSubClass', 'LotFrontage', 'LotArea', 'YearBuilt', '1stFlrSF', 'FullBath', 'BedroomAbvGr',
            'KitchenAbvGr', 'GarageArea']
test_X = test_data[test_features]
test_X = np.nan_to_num(test_X)
test_preds = rf_model.predict(test_X)

# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)