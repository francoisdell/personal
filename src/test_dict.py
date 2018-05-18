import numpy as np
lol = {'a': np.array(['derp','wee'])}

hah = {'a': np.array(['derp','wee'])}

print(lol)
# print(lol==hah)
print(np.testing.assert_equal(lol, hah))