import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pd.set_option("display.max_rows", None, "display.max_columns", None)

melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data = melbourne_data.dropna(axis=0)

y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

melbourne_model = RandomForestRegressor(random_state=1)
melbourne_model.fit(train_X, train_y)
melbs_preds = melbourne_model.predict(val_X)
mae = mean_absolute_error(val_y, melbs_preds)
print(mae)

cols = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Alley', 'LotShape', '']