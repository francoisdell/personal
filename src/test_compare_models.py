from sklearn.linear_model import LogisticRegression

p1 = LogisticRegression().get_params()
p2 = LogisticRegression().get_params()
print(p1)
print(p1==p2)