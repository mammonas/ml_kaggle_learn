import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

pd.set_option("display.max_rows", None, "display.max_columns", None)

train_file_path = './home-data-for-ml-course/train.csv'
train_data = pd.read_csv(train_file_path)

train_y = train_data.SalePrice
numeric_train_data = train_data.select_dtypes(include=np.number).drop(labels=['SalePrice'], axis=1)

nan_features = []
for col in numeric_train_data.columns:
    count_null = numeric_train_data[col].isnull().sum()
    if count_null > 0:
        nan_features.append(col)


numeric_train_data = numeric_train_data.drop(labels=nan_features, axis=1)
print(numeric_train_data.columns)
s_train_X, s_val_X, s_train_y, s_val_y = train_test_split(numeric_train_data, train_y, random_state=0)
rf_model = RandomForestRegressor(random_state=1)
# rf_model.fit(s_train_X, s_train_y)
rf_model.fit(numeric_train_data, train_y)
predicts = rf_model.predict(s_val_X)
mae = mean_absolute_error(predicts, s_val_y)
print('MAE %i' % mae)

test_file_path = './home-data-for-ml-course/test.csv'
test_data = pd.read_csv(test_file_path)

test_X = test_data[numeric_train_data.columns]
test_X = test_X.fillna(test_X.mean())

test_preds = rf_model.predict(test_X)
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)