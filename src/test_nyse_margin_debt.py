import requests
import datetime
import pandas as pd
import io

def get_nyse_margin_debt() -> pd.Series:
    url = 'http://www.nyxdata.com/nysedata/asp/factbook/table_export_csv.asp?mode=tables&key=50'
    with requests.Session() as s:
        download = s.get(url=url)

    strio = io.StringIO(download.text)
    df = pd.read_table(strio, sep='\\t', skiprows=3)

    print(df)

    df['End of month'] = pd.DatetimeIndex(pd.to_datetime(df['End of month']),
                                          dtype=datetime.date).to_period('M').to_datetime('M')
    df.set_index(['End of month'], inplace=True, drop=True)
    print(df)

lol = get_nyse_margin_debt()
print(lol)