from datetime import datetime, timedelta
from typing import List


def date_range(start_date: str, end_date: str) -> List[str]:
    """
    返回从开始日期到结束日期的每一天的日期列表
    
    参数:
    start_date (str): 开始日期，格式为 "yyyy-mm-dd"
    end_date (str): 结束日期，格式为 "yyyy-mm-dd"
    
    返回:
    List[str]: 包含日期字符串的列表，格式为 "yyyy-mm-dd"
    """
    # 将输入的日期字符串转换为datetime对象
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # 计算日期差
    delta = end - start
    
    # 生成日期列表
    dates = []
    for i in range(delta.days + 1):
        date = start + timedelta(days=i)
        dates.append(date.strftime("%Y-%m-%d"))
    
    return dates


# 示例用法
if __name__ == "__main__":
    # 测试函数
    start = "2023-01-01"
    end = "2023-01-05"
    result = date_range(start, end)
    print(f"从 {start} 到 {end} 的日期列表:")
    for date in result:
        print(date)