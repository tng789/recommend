'''用于开发调试， 手工运行'''

from selector import Selector
from datetime import datetime   

from baostock_ops import BaostockOps
def main():

    program = Selector()

    today = datetime.now().strftime("%Y-%m-%d")
    BaostockOps().update_dataset()

    # ch = input("press enter to continue.....")
    for stock_pool in (["zz500", "zz1000"]):
        df = program.make_dataframe(stock_pool=stock_pool)
        program.predict(stock_pool=stock_pool, df_predict=df, val_end=today)

if __name__ == "__main__":
    main()