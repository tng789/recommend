import baostock as bs
import pandas as pd

from datetime import datetime, timedelta
from pathlib import Path

class BaostockOps:
    def __init__(self, working_dir=".working", base_dir=".local"):
        self.working_dir = Path(working_dir)
        self.base_dir = Path(base_dir)
        self.very_beginning = "2020-01-01"
        self.calendar = self.load_calendar()

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

            print(f"the last date of {code} ohlcv: {df.iloc[-1]['date']}")
        else:
            print(f"no new ohlcv data for {code}")

        return df

    def update_stock_list(self, stock_pool:str)->list[str]:
        '''self.working_dir路径下取得所有csv的文件名，不含扩展名，写入列表变量，返回该列表。'''
        csv_files = []
        if stock_pool == "":
            for file_path in self.working_dir.glob("*.csv"):
                csv_files.append(file_path.stem)  # stem属性是文件名不包含扩展名
        else:
            df = pd.read_csv(self.base_dir/stock_pool/f"{stock_pool}_stocks.csv")
            csv_files = df['code'].tolist()
        return csv_files
    #---------------------------
    # 更新股票数据，增量方式更新，不需要整个下载全部数据
    #----------------------------
    def _update_stock_data(self, code:str)->None:

        self.working_dir.mkdir(parents=True, exist_ok=True)

        # 获取股票历史数据,如果本地磁盘没有，则直接从baostock获取；若有，则取本地磁盘数据的最后一天的下一天，以此为起始日。截止日均为今日。
        datafile = self.working_dir/f"{code}.csv"
        today = datetime.now().strftime("%Y-%m-%d")
        if not  datafile.exists():
            df = pd.DataFrame()
            start_date = "2000-01-01"
        else:
            df = pd.read_csv(datafile,parse_dates=True,skip_blank_lines=True)
            df = df.dropna(how='all')

            df['date'] = df['date'].str.replace("/","-")

            last_date = df.iloc[-1]['date']
            start_date = datetime.strptime(last_date,"%Y-%m-%d") + timedelta(days=1)
            start_date = start_date.strftime("%Y-%m-%d")

        #取出之后合并
        end_date = today
        if start_date > end_date:
            new_transaction = pd.DataFrame()
        else:
            new_transaction = self._fetch_stocks(code, start_date, end_date)
            # days = new_transaction.shape[0]

        # 没有取到数据，则返回原先的数据, 有就拼接起来
        if new_transaction.shape[0] == 0:
            return df

        df = pd.concat([df, new_transaction], axis=0).drop_duplicates()
        # 重置索引
        df.reset_index(drop=True, inplace=True)
        df.to_csv(datafile, index=False, date_format='%Y-%m-%d',encoding="gbk")

        return df
    def update_dataset(self, stock_pool:str="")->None:

        stock_list = self.update_stock_list(stock_pool)

        entry = bs.login()          #一次登录，取多条数据
        if entry.error_code != '0':
            print(entry.error_msg)
            bs.logout()
            return

        for code in stock_list:
            self._update_stock_data(code)
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

        df_calendar.to_csv(calendar, index=False)
        return df_calendar
