import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pd.set_option("display.max_rows", None, "display.max_columns", None)

melbourne_file_path = './melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

print(melbourne_data.describe())
print(melbourne_data.columns)

melbourne_data = melbourne_data.dropna(axis=0)

y = melbourne_data.Price
print(y)

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X.describe())
print(X.head())

melbourne_model = DecisionTreeRegressor(random_state=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
melbourne_model.fit(train_X, train_y)

# print('Predict for just %s:' % len(X.head()))
# print(X.head())
# predict = melbourne_model.predict(X.head())
# print('Predictions are:')
# print(predict)

predict = melbourne_model.predict(val_X)

mae = mean_absolute_error(val_y, predict)
print('Mean Absolute Error')
print(mae)
