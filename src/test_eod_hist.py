from pandas.compat import StringIO
import requests_cache
import pandas as pd
from datetime import timedelta
import requests

code = 'NYA.INDX'
api_token = open('token_eodhist.txt', mode='r').read()
url = 'https://eodhistoricaldata.com/api/eod/{0}?api_token={1}'.format(code, api_token)
params = {'api_token': api_token}
expire_after = timedelta(days=1).total_seconds()
session = requests_cache.CachedSession(cache_name='cache', backend='sqlite', expire_after=expire_after)
r = session.get(url, params=params)
if r.status_code != requests.codes.ok:
    session = requests.Session()
    r = session.get(url, params=params)
if r.status_code == requests.codes.ok:
    df = pd.read_csv(StringIO(r.text), skipfooter=1, parse_dates=[0], index_col=0, engine='python')
    data = df['Close']
else:
    raise Exception(r.status_code, r.reason, url)

print(data.head())
print(data.tail())
