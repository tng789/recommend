import baostock as bs
import pandas as pd

from datetime import datetime , timedelta
from pathlib import Path

class BaostockOps:
    index_mapping = {
        "hs300": "sh.000300",
        "zz500": "sh.000905",
        "zz1000": "sh.000852"
    }
    def __init__(self, home="."):
        self.home = Path(home).resolve()
        self.working_dir = self.home / "working"
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        self.base_dir = self.home / "local"
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        
        self.very_beginning = "2020-01-01"
        self.calendar = self.load_calendar()

        # 读取base_dir及其子目录中所有匹配total_*.parquet的文件
        parquet_files = list(self.base_dir.rglob("total_*.parquet"))
        if not parquet_files:
            # 如果在子目录中没有找到，尝试在当前目录查找
            parquet_files = list(self.base_dir.glob("total_*.parquet"))
        
        if parquet_files:
            # 读取所有匹配的parquet文件并合并
            dfs = [pd.read_parquet(file) for file in parquet_files]
            self.total_dataset = pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError(f"No parquet files matching 'total_*.parquet' found in {self.base_dir} or its subdirectories")

        self.total_dataset = self.total_dataset.set_index('date')
    def _convert_to_float(self, df:pd.DataFrame)->pd.DataFrame:
        df = df.mask(df == "", 0)
        for col in df.columns:
            if col not in ['date', 'code']:
                df[col] = df[col].astype(float)
        return df
    def _fetch_stocks(self, code:str, start_date:str, end_date:str, freq = 'd')->pd.DataFrame:

        cols = ",".join(['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'turn','peTTM','psTTM','pcfNcfTTM','pbMRQ'])
        empty_df = pd.DataFrame()

        rs = bs.query_history_k_data_plus(
                code,
                cols,
                start_date = start_date,
                end_date   = end_date,
                frequency  = freq,
                adjustflag= "2"      #复权类型，默认不复权：3；1：后复权；2：前复权。 固定不变。
        )

        if rs.error_code != '0':
            print('query_history_k_data_plus respond error_msg:'+rs.error_msg)
            return empty_df

        data_list = []
        while rs.next():
            # 获取一条记录，将记录合并在一起
            bs_data = rs.get_row_data()
            data_list.append(bs_data)

        df = pd.DataFrame(data_list, columns=rs.fields)
        if df.shape[0] != 0:
            # 删去成交量为零的行，重置索引
            df = self._convert_to_float(df)
            # df = df.replace(0,np.nan).dropna()
            df.reset_index(drop=True, inplace=True)

            df.sort_values(by=['date'], ascending=True, inplace=True)

        else:
            print(f"no new ohlcv data for {code}")

        return df

    def _fetch_index(self, code:str, start_date:str, end_date:str, freq = 'd')->pd.DataFrame:

        cols = ",".join(['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'pctChg'])
        empty_df = pd.DataFrame()

        rs = bs.query_history_k_data_plus(          # 指数， 与股票不同
                code,
                cols,
                start_date = start_date,
                end_date   = end_date,
                frequency  = freq,
        )

        if rs.error_code != '0':
            print('query_history_k_data_plus respond error_msg:'+rs.error_msg)
            return empty_df

        data_list = []
        while rs.next():
            # 获取一条记录，将记录合并在一起
            bs_data = rs.get_row_data()
            data_list.append(bs_data)

        df = pd.DataFrame(data_list, columns=rs.fields)
        if df.shape[0] != 0:
            # 删去成交量为零的行，重置索引
            df = self._convert_to_float(df)
            # df = df.replace(0,np.nan).dropna()
            df.reset_index(drop=True, inplace=True)

            df.sort_values(by=['date'], ascending=True, inplace=True)

            print(f"the last date of {code} ohlcv: {df.iloc[-1]['date']}")
        else:
            print(f"no new ohlcv data for {code}")

        return df
    def save_parquet(self, df:pd.DataFrame, date_col:str="date")->None:
        df[date_col] = pd.to_datetime(df[date_col])
        
        # 筛选2020-01-01之后的数据
        df_filtered = df[df[date_col] >= self.very_beginning]
        
        # 检查是否有符合条件的数据
        if df_filtered.empty:
            print("没有找到2020-01-01之后的数据")
        else:
            print(f"找到 {len(df_filtered)} 条2020-01-01之后的数据")
            
            # 按年份分组并保存到不同的parquet文件
            for year, group in df_filtered.groupby(df_filtered[date_col].dt.year):
                output_file = Path(self.base_dir) / f"total_{year}.parquet"
                group.to_parquet(output_file)
                print(f"Saved {output_file} with {len(group)} rows")
                
    def update_dataset(self, refresh:bool=True )->None:

        entry = bs.login()          #一次登录，取多条数据
        if entry.error_code != '0':
            print(entry.error_msg)
            bs.logout()
            return

        today = datetime.now()
        today_str = datetime.strftime(today,'%Y-%m-%d')
        
        if refresh:                         # 刷新，从2020-01-01开始
            last_day_str = self.very_beginning
        else:
            last_day = self.total_dataset.index.max()
            last_day_str = datetime.strftime(last_day,'%Y-%m-%d')
        
        if last_day_str >= today_str:
            print(f"库中最后一天是 {datetime.strftime(last_day,'%Y-%m-%d')}, 无需更新")
            return
            
        # 从csi300,csi500,csi1000等三个文件中读取股票代码列表，按照股票代码从baostock下载数据，并保存在parquet文件中。
        stock_list = []
        for stock_list_file in (self.home / "csi300_list.csv", self.home / "csi500_list.csv", self.home / "csi1000_list.csv"):
            df = pd.read_csv(stock_list_file)
            stocks = df['code'].tolist()
            stock_list = stock_list + stocks

        exisiting_stocks = self.total_dataset['code'].unique().tolist()

        # 收集所有待合并的dataframes，然后一次性合并
        dataframes_to_concat = []
        for code in stock_list:
            # 如果该股票代码没有数据，则下载该股票代码从very_beginning开始的数据

            if code not in exisiting_stocks:
                last_day_str = self.very_beginning
            else:
                last_day = self.total_dataset.index.max()
                last_day_str = datetime.strftime(last_day,"%Y-%m-%d")

            results = self._fetch_stocks(code, last_day_str, today_str)
            dataframes_to_concat.append(results)
            print(f"{code}  ohlcv data read")#  from {last_day} to {datetime.strftime(today,'%Y-%m-%d')} pulled")

        # 一次性合并所有数据框，然后进行去重
        if dataframes_to_concat:  # 确保有数据才进行合并
            new_data = pd.concat(dataframes_to_concat, axis=0, ignore_index=True)
            self.total_dataset = pd.concat([self.total_dataset, new_data], axis=0, ignore_index=True).drop_duplicates()

        self.save_parquet(self.total_dataset)

        bs.logout()

    def update_index(self)->None:
        today = datetime.now()
        # last_year = today.year - 1
        end_date = today.strftime("%Y-%m-%d")
        start_date = self.very_beginning

#        self.index_mapping = {
#            "上证指数": "sh.000001",
#            "深证成指": "sz.399001",
#            "沪深300": "sh.000300",
#            "中证1000": "sz.399005",
#            "中证500": "sz.399010",
#            "中证100": "sz.399106",
#            "中小板指": "sz.399399",
#            "创业板指": "sz.399283",
#            "上证50": "sh.000016",
#        }

        bs.login()
        for name, code in self.index_mapping.items():
            df  = self._fetch_index(code,start_date,end_date)
            # print(df.head())
            df.to_csv(self.base_dir / f"{code}.csv", index=False)
        bs.logout()
    def load_calendar(self, end_date:str=""):
        # 登录
        if end_date == "":
            end_date = datetime.now().strftime("%Y-%m-%d")

        def get_trading_days(start_date:str, end_date:str="")->pd.DataFrame:
            rs = bs.login()
            if rs.error_code != '0':
                print(f"log into baostock: {rs.error_msg}")
                return pd.DataFrame()

            rs = bs.query_trade_dates(start_date=start_date,  end_date=end_date)
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            df = pd.DataFrame(data_list, columns=rs.fields)
            bs.logout()

            return df

        calendar = self.base_dir / "calendar.csv"
        start_date = self.very_beginning
        if calendar.exists():
            df_calendar =  pd.read_csv(calendar, parse_dates=True)
            last_date = df_calendar['calendar_date'].max()
            if last_date == end_date:
                return df_calendar
            else:
                df_calendar = get_trading_days(start_date, end_date)
        else:
            df_calendar = get_trading_days(start_date, end_date)

        df_calendar['calendar_date'] = pd.to_datetime(df_calendar['calendar_date'])
        df_calendar.to_csv(calendar, index=False)
        return df_calendar

    #---
    # 判断今日是否为交易日
    #---
    def is_trading_day(self, day:str)->bool:
        '''从self.calendar取得day所对应的is_trading_day的值，为1返回True，否则返回False
        '''
        today = datetime.now().strftime("%Y-%m-%d")

        if day == "":
            day = today

        if day > today or day < "2020-01-01":
            return False

        # self.calendar['calendar_date']
        trading = self.calendar.loc[self.calendar['calendar_date']==day]['is_trading_day']
        return trading.all() == 1
    
def last_day_today(day:datetime)->bool:
       # 计算明天的日期
    tomorrow = day + timedelta(days=1)
    
    # 如果明天的月份与今天不同，则今天是本月最后一天
    return tomorrow.month != day.month

#for testing
if __name__ == "__main__":
    ops = BaostockOps()
    today = datetime.now()
    today_str = datetime.strftime(today,'%Y-%m-%d')
    # 判断今天是不是每月的最后一天，如是，则更新dataset和index，且初始日期为2020-01-01,截止日期为今天。
    # 如果不是每月最后一天，再判断今天是否交易日，如是，则update dataset 和 index，初始日期为parquet文件中的最后一天(默认)，截止日期为今天。
    # 如果不是交易日，则不更新。
    #
    if last_day_today(today):
        print(f"{today_str} 是本月最后一日，更新全部数据")
        ops.update_dataset(refresh = True)
        ops.update_index()
    elif ops.is_trading_day(today_str):
        ops.update_dataset()
        ops.update_index()
    else:
        print(f"{today_str} 不是交易日，不更新数据")
        # pass
