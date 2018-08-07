import numpy as np
lol = {'a': np.array(['derp','wee'])}

hah = {'a': np.array(['derp','wee'])}

print(lol)
# print(lol==hah)
print(np.testing.assert_equal(lol, hah))
d = dict.fromkeys(('a', 100),('b',200))
print(d)

d = dict((('method', 'corr'), ('limit', 0.99)))
print(d)

d = dict({('a', 100)})
print(d)