import pandas as pd
from matplotlib import pyplot as ml
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"C:\Users\Omistaja\Documents\statistic_id1099351_gold-medal-times-in-the-marathon-at-the-summer-olympics-1896-2020.csv", sep=";")
#data has also womens winning times as well but the since Olympic marathon has been in womens Olympics from 1984 there is lot less data
#Let's make scatterplot to see if there is correlation between womens and mens times
ml.scatter(df["Winning time in minutes"],df["Women"])
#ml.xlabel("Men's winning time")
#ml.ylabel("Women's winning time")
#ml.show()
#from the plot it seems there is eather a very weak correlation or no correlation at all
#for this reason I'll drop the womens times from data
df.drop(["Women"],axis=1, inplace=True)
print(df.head(5))
#ml.plot(df["Year"],df["Winning time in minutes"])
#ml.title("Winning times against year plot")
#ml.xlabel("Year")
#ml.ylabel("Winning time(minutes)")
#Since St. Louis 1904 was outliar, we will drop it
#Reasoning for the dropping
df.drop([2],axis=0,inplace=True)
#ml.plot(df["Year"],df["Winning time in minutes"])
#ml.title("Winning times against year, exception shown in different color")
#ml.show()
#From the plot it's easy to see that there is clear negative correlation between the year and the winning time
#To be sure of the I'll calculate the correlation between these two
print(np.corrcoef(df["Winning time in minutes"],df["Year"])[0,1])
# Correlation in -0.9211... so it's clear that there is a strong correlation between the variables

#For this project polynomical regression is closen, so let's orgranize the data for that
# decrease of the polynomial regression will bw decided after trial
X=df["Year"].to_numpy().reshape(-1,1)
y=df["Winning time in minutes"].to_numpy()
#X_train, X_val, y_train, y_val=train_test_split(X,y, test_size=0.5, random_state=22)
x=0
mean=0
while x <20:
    cv=KFold(n_splits=5, random_state=x, shuffle=True)
    degrees=[2,4,7,10]
    training_error=[]
    validation_error=[]
    ml.figure()
    n=0
    for i, degree in enumerate(degrees):
        ml.subplot(2,2,i+1)
        val_er=[]
        for train_index, index in cv.split(y):
            X_train, X_val = X[train_index], X[index]
            y_train, y_val = y[train_index], y[index]

            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train)

            lin_regr = LinearRegression(fit_intercept=False)

            lin_regr.fit(X_train_poly, y_train)

            X_val_poly = poly.fit_transform(X_val)
            y_pred_val = lin_regr.predict(X_val_poly)
            val_erro = mean_squared_error(y_val, y_pred_val)
            val_er.append(val_erro)
        """
        This part of the code was 
        
        poly=PolynomialFeatures(degree=degree)
        X_train_poly=poly.fit_transform(X_train)
        lin_regr.fit(X_train_poly,y_train)
    
        y_pred_train=lin_regr.predict(X_train_poly)
        tr_error = mean_squared_error(y_train, y_pred_train)
        X_val_poly = poly.fit_transform(X_val)
        y_pred_val = lin_regr.predict(X_val_poly)
        val_error = mean_squared_error(y_val, y_pred_val)"""
        val_error=sum(val_er)/len(val_er)
        #training_error.append(tr_error)
        validation_error.append(val_error)
        ml.plot(X, lin_regr.predict(poly.transform(X.reshape(-1, 1))))
        ml.scatter(X, y, color="r")
        ml.title(f"polynomial degree {degree}", loc="center")
        n+=1
    ml.show()
    print(validation_error)


    ml.plot(degrees, validation_error, label="Validation error")
    #ml.plot(degrees,training_error,label="Training error")
    ml.legend(loc='best')
    ml.xlabel("Degrees")
    ml.ylabel("Error")
    ml.show()
    mean+=validation_error[0]
    x+=1
mean=mean/20
print(mean)

