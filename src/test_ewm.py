import pandas as pd
s = pd.Series([1,2,3,4,5,6,7,8,9,10])
ewma = s.ewm(alpha=0.125).mean()
print(ewma)