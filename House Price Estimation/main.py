import joblib
import numpy as np

model = joblib.load('regression_model.pkl')

long= -122
lat= 40
house_age= 3
room= 4
bedroom= 22
pop= 3
household= 6
income= 9
ocean=1
inland=0
island=0
near_bay =0
nocean =  0

test = [long,lat,house_age,room,bedroom,pop,household,income,ocean,inland,island,near_bay,nocean]
test.append(test[3]/test[2])
test.append(test[2]/test[5])
test = np.array(test).reshape(1,-1)

pred = model.predict(test)


print(pred)
