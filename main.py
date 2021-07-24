import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pandas.read_csv("covid-spread-cases.csv")

X = pandas.DataFrame(data, columns=['first_week'])
y = pandas.DataFrame(data, columns=['second_week'])

plt.scatter(X, y, alpha=0.3)
plt.xlabel("First Week cases")
plt.title("Covid-19 Rate Indonesia, USA, India")
plt.ylabel("Second week cases")

# Show chart and the actual point
regression = LinearRegression()
regression.fit(X, y)
plt.plot(X, regression.predict(X))
print(regression.coef_)
print(regression.intercept_)
# print(regression.predict(X))
print(f"3 Minggu yang akan datang jumlah prediksi covid akan mencapai: {regression.intercept_[0] + (regression.coef_[0][0] * 309015.0)}")
plt.show()





