import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

train_file_path = 'home-data-for-ml-course/train.csv'
train_data = pd.read_csv(train_file_path)
# print(train_data.head())

y = train_data.SalePrice
features = train_data.select_dtypes(include='number').columns.drop(['SalePrice'])
X = train_data[features]
# print(X.head())

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
missing_values_cols = [col for col in X_train.columns if X_train[col].isnull().any()]
X_train = X_train.drop(missing_values_cols, axis=1)
X_val = X_val.drop(missing_values_cols, axis=1)
# print(X_train.head())

# https://www.kaggle.com/mammonas/exercise-introduction/edit
# https://www.kaggle.com/alexisbcook/missing-values