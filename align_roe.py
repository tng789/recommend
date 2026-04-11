import pandas as pd




def merge_with_backward_fill(A, B):
    """
    将B中的data列合并到A中，使用向前填充方式
    
    Parameters:
    A: DataFrame with columns ['code', 'date', ...]
    B: DataFrame with columns ['code', 'date', 'data']
    
    Returns:
    DataFrame: A with new column 'data' from B, forward filled
    """
    
    # 确保date列是datetime类型
    A = A.copy()
    B = B.copy()
    
    A['date'] = pd.to_datetime(A['date'])
    B['date'] = pd.to_datetime(B['pubDate'])
    
    # 按code和date排序，这是merge_asof要求的
    A = A.sort_values(['code', 'date'])
    B = B[['date','code','roeAvg']].sort_values(['code', 'date'])
    
    # 使用merge_asof进行向前填充合并
    # direction='forward'表示使用B中最近的未来日期来填充
    result = pd.merge_asof(
        A, 
        B, 
        on='date', 
        by='code',  # 按code分组匹配
        direction='backward',  # 向前填充
        allow_exact_matches=True  # 允许精确匹配
    )
    
    return result

def main():
    df = pd.read_csv(".working/sh.600028.csv")
    df = df[df['date']>="2024-01-01"]

    roe = pd.read_csv(".local/roe.csv")
    roe = roe[roe['code'] == "sh.600028"]
    result = merge_with_backward_fill(df, roe)   

    print(result.head())

    result.to_csv("roe_aligned.csv", index=False)
if __name__ == "__main__":
    main()
