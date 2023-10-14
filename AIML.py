import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Step 1
df = pd.read_csv("/kaggle/input/salary-dataset-simple-linear-regression/Salary_dataset.csv")

#Step 2
plt.scatter(df['YearsExperience'], df['Salary'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.show()

#Step 3
X = df[['YearsExperience']]
y = df['Salary']

#Step 4 
model = LinearRegression()
model.fit(X,y) # calculates the best fitting relation between x and y 
slope = model.coef_[0] # represents coefficient of x in y = mx+b
intercept = model.intercept_ # point where regression line crosses y axis 

#Step 5 
def linear_regression(x):
    return slope * x + intercept

#Step 6
plt.scatter(df['YearsExperience'],df['Salary'],label = 'Data Points')
plt.plot(df['YearsExperience'],[linear_regression(xi) for xi in df['YearsExperience']], color = 'red' , label = 'Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linear Regression')
plt.show()

#Step 7
new_x = 6 
prediction = linear_regression(new_x)
print(f"Prediction for x = {new_x} : {prediction}")

new_x = pd.DataFrame({'YearsExperience' : [6]})
print(model.predict(new_x))