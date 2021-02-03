import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def score_model(md, X_t, X_v, y_t, y_v):
    md.fit(X_t, y_t)
    predictions = md.predict(X_v)
    return mean_absolute_error(y_v, predictions)


def get_best_model(mds, X_t, X_v, y_t, y_v):
    best_model = None
    lowest_mea = None
    for i in range(0, len(mds)):
        mea = score_model(mds[i], X_t, X_v, y_t, y_v)
        print("MEA is %d" % mea)
        if i == 0:
            best_model = mds[i]
            lowest_mea = mea
        else:
            if lowest_mea > mea:
                lowest_mea = mea
                best_model = mds[i]
    print("Best MEA is %d of model %s" % (lowest_mea, best_model))
    return best_model


def tuning_model(model, X_train, X_val, y_train, y_val):
    lowest_score = None
    best_n_estimators = None
    for i in range(10, 300, 20):
        model.n_estimators = i
        score = score_model(model, X_train, X_val, y_train, y_val)
        print("score for n_estimators %d is %d" % (i, score))
        if i == 10:
            lowest_score = score
            best_n_estimators = i
        else:
            if lowest_score > score:
                lowest_score = score
                best_n_estimators = i

    model.n_estimators = best_n_estimators
    print("Best model after tunning is %s" % model)
    return model


train_file_path = 'home-data-for-ml-course/train.csv'
train_data = pd.read_csv(train_file_path)

features_columns = train_data.columns.drop(['SalePrice'])
y = train_data.SalePrice
X = train_data[features_columns].select_dtypes(include='number')


X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
missing_values_cols = [col for col in X_train.columns if X_train[col].isnull().any()]
X_train = X_train.drop(missing_values_cols, axis=1)
X_val = X_val.drop(missing_values_cols, axis=1)


models = [
    RandomForestRegressor(n_estimators=50, random_state=0),
    RandomForestRegressor(n_estimators=100, random_state=0),
    RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0),
    RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0),
    RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)
]

model = get_best_model(models, X_train, X_val, y_train, y_val)
model = tuning_model(model, X_train, X_val, y_train, y_val)

X_train_full = X.drop(missing_values_cols, axis=1)
model.fit(X_train_full, y)

test_file_path = 'home-data-for-ml-course/test.csv'
test_data = pd.read_csv(test_file_path)

X_test = test_data[X_train.columns]
X_test = X_test.fillna(X_test.mean())

preds = model.predict(X_test)
output = pd.DataFrame({
    'Id': X_test.Id,
    'SalePrice': preds
})
output.to_csv('submission_17564.csv', index=False)

# https://www.kaggle.com/mammonas/exercise-introduction/edit
# https://www.kaggle.com/alexisbcook/missing-values