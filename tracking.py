import pandas as pd
import sys
from stockdata_ops import stock_data

# import numpy as np
def convert_to_wide_format(df_list, code_names):
    """
    使用pivot方法实现相同的功能
    """
    # 为每个DataFrame添加code列
    for df, code in zip(df_list, code_names):
        df['code'] = code
    
    # 合并所有DataFrame
    combined = pd.concat(df_list, ignore_index=True)
    
    # 使用pivot转换
    result = combined.pivot(index='code', columns='date', values='close')
    
    # 重置索引
    result.reset_index(inplace=True)
    
    return result
def main():
    portfolio = sys.argv[1]         #"CSI1000"
    date = sys.argv[2]              #"2026-03-04"

    database = stock_data()
    print(f"portfolio: {portfolio}")
    
    df = pd.read_csv(f"local/{portfolio}/predictions.csv", index_col='date')
    top10 = df.loc[df.index == date]['code'].to_list()

    if len(top10) == 0:
        print("No stock is there in the prediction list")
        return

    prediction_list = []
    codes = []
    for stock in top10:
        df_stock = database.total_dataset.copy()              #  pd.read_csv(f".working/{stock}.csv")
        # df_prices = df_stock[df_stock['date'] >= date]
        df_prices = df_stock[df_stock.index >= date]
        df_prices =df_prices.iloc[:11]
        # df_prices = df_prices[['date', 'close']]

        print(df_prices)

        df_prices.reset_index(inplace=True)
        # df_prices = df_prices.T
        prediction_list.append(df_prices)
        codes.append(stock)
    
    predictions = convert_to_wide_format(prediction_list, codes)

    predictions.to_csv(f"local/{portfolio}/prediction_tracking_{date}.csv", index=False)    

if __name__ == "__main__":
    main()

