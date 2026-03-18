from selector import Selector
from datetime import datetime   
def main():

    program = Selector()

    today = datetime.now().strftime("%Y-%m-%d")
    program.update_dataset()

    # ch = input("press enter to continue.....")
    for stock_pool in (["zz500", "zz1000"]):
        df = program.make_dataframe(stock_pool=stock_pool)
        program.predict(stock_pool=stock_pool, df_predict=df, val_end=today)

if __name__ == "__main__":
    main()