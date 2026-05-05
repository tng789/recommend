# easy_screener 回测

import pandas as pd
# from baostock_ops import BaostockOps
from stockdata_ops import stock_data
# from datetime import datetime, timedelta
# from pathlib import Path

if __name__ == "__main__":

    database = stock_data()
    
#    csi300_list = database.stock_map['CSI300']
#    csi500_list = database.stock_map['CSI500']
#    csi1000_list = database.stock_map['CSI1000']
#    
#    all_stocks_list = csi500_list #+ csi500_list + csi1000_list

    for pool, stock_list in database.stock_map.items():
    
        # 使用向量化操作一次性过滤所有需要的股票数据
        mask = database.total_dataset['code'].isin(stock_list)
        filtered_data = database.total_dataset[mask]
    
        # 按股票代码分组，创建字典
        all_stocks_data = {}
        for stock_code, group in filtered_data.groupby('code'):
            df = group.copy()
            
            # 注意：total_dataset已经将date设为索引，所以这里不再重复设置或重置
            # 确保索引是 DatetimeIndex 以符合后续计算因子的要求
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            all_stocks_data[stock_code] = df
        
        picks = pd.read_csv( database.base_dir / "picks.csv")
        days = picks['date'].unique().tolist()
    
        # 从picks读取某一天(date,比如2026-01-02)的csi300\csi500\csi1000的股票列表
        # 再根据列表获取从当天到之后的30天的每个股票close列的数据，即， 如sh.600118的2026-01-02开始30天的close列数据 
        # 再计算每一天与当天的close相比的涨跌幅，写在另一列中。
        # 按照date保存到working_dir/picks_returns_{date}.csv中 
        
        # 遍历每个唯一的日期
        for target_date in days:
            # 获取指定日期的股票列表
            daily_picks = picks[picks['date'] == target_date]
            stock_list = daily_picks['code'].tolist()
            
            # 初始化结果DataFrame
            result_data = []
            
            # 遍历每只股票
            for stock_code in stock_list:
                if stock_code in all_stocks_data:
                    # 获取这只股票的数据
                    stock_data = all_stocks_data[stock_code]
                    
                    # 找到目标日期及其后60天的数据
                    target_date_obj = pd.to_datetime(target_date)
                    end_date_obj = target_date_obj + pd.Timedelta(days=60)
                    
                    # 筛选这段时间的数据
                    date_mask = (stock_data.index >= target_date_obj) & (stock_data.index <= end_date_obj)
                    period_data = stock_data[date_mask]
                    
                    if not period_data.empty:
                        # 获取基准价格（目标日期的价格）
                        base_close = period_data.iloc[0]['close'] if len(period_data) > 0 else None
                        
                        if base_close is not None and base_close != 0:
                            # 计算相对于基准日期的涨跌幅
                            period_data = period_data.copy()
                            period_data['pct_change_from_start'] = (period_data['close'] - base_close) / base_close
                            
                            # 添加股票代码列
                            period_data['code'] = stock_code
                            
                            # 只保留需要的列
                            period_result = period_data[['code', 'close', 'pct_change_from_start']].copy()
                            
                            # 重置索引以保留日期信息
                            period_result.reset_index(inplace=True)
                            
                            result_data.append(period_result)
            
            # 合并所有股票的结果
            if result_data:
                final_result = pd.concat(result_data, ignore_index=True)
                
                # 按日期和股票代码排序
                final_result.sort_values(['date', 'code'], inplace=True)
                
                # 保存到文件
                output_dir = database.working_dir / pool
                output_dir.mkdir(parents=True, exist_ok=True)
                output_filename = output_dir / f"picks_returns_{target_date}.csv"
                final_result.to_csv(output_filename, index=False)
                print(f"已保存 {target_date} 的回测数据到 {output_filename}")
            else:
                print(f"警告: 日期 {target_date} 没有找到有效的股票数据")
    