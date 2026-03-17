import baostock as bs
import pandas as pd
from datetime import datetime
import sys
def get_roe(code, years) :
    # 查询季频估值指标盈利能力
    roe = pd.DataFrame()
    profit_list = []
    for year in years:
        for q in (1,2,3,4):                     # 1-4季度
            rs_profit = bs.query_profit_data(code=code, year=year, quarter=q)
            while (rs_profit.error_code == '0') & rs_profit.next():
                profit_list.append(rs_profit.get_row_data())
    roe = pd.DataFrame(profit_list, columns=rs_profit.fields)
    # roe = pd.concat([roe, result_profit],axis=0)
    return roe
def main():
    portfolio = sys.argv[1]
    # portfolio = "sz50"
    df = pd.read_csv(f".local/{portfolio}/{portfolio}_stocks.csv")
    stock_list = df["code"].tolist()
    
    today = datetime.now()
    years = [today.year-2, today.year-1,today.year]
    # month = [today.month]

    
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    consolidated_roe = pd.DataFrame()
    for stock in stock_list:
        roe = get_roe(stock, years)
        consolidated_roe = pd.concat([consolidated_roe, roe],axis=0)
    consolidated_roe.to_csv(f".local/{portfolio}/roe_{portfolio}.csv", encoding="gbk", index=False)
    bs.logout()

if __name__ == "__main__":
    main()