import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)

melbourne_file_path = './melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.describe())
