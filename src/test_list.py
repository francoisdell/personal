lol = 'abcde'
hah = ['a', 'e', 'f', ['b', 'c', 'd']]

print(hah[-3:])

lol1 = list(lol)
hah1 = list(hah)

print(lol1)
print(hah1)

for l in lol1:
    print(l)
for l in hah1:
    print(l)

derp = lol1.pop()
print(derp)
print(lol1)

hehe = [(v, 'stack') for v in list(hah1.pop())]
print(hah1)
print(hehe)


