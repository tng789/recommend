from selector import Selector
import pandas as pd
from pathlib import Path
import baostock  as bs
s = Selector()

zz1000_file = Path(".local/zz1000/zz1000_stocks.csv")
df = pd.read_csv(zz1000_file, index_col=None)
stocks = df['code'].tolist()

bs.login()
for code in stocks:
    s._update_stock_data(code)
bs.logout()    