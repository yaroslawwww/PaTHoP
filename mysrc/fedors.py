from mysrc.Research import rmse
from mysrc.TimeSeries import TimeSeries

ts = TimeSeries("Lorentz", size=100000, r=28, dt=0.01)

x = ts.values
print(rmse(x,[0.5] * len(x)))
