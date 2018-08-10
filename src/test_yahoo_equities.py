# #!/usr/bin/python
# import yahoo_finance
# import pandas as pd
#
# symbol = yahoo_finance.Share("^NYA")
# data = symbol.get_historical("1960-01-01", "2016-06-30")
# df = pd.DataFrame(data)
#
# # Output data into CSV
# print(df)

from pandas_datareader import data as dreader
symbols = ['^NYA']

pnls = {i: dreader.DataReader(i, 'fred', '1960-01-01', '2016-09-01') for i in symbols}

for df_name in pnls:
    pnls.get(df_name).to_csv("{}_data.csv".format(df_name), index=True, header=True)
