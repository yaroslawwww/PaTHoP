def main():

    from Predictions import *

    from sklearn.metrics import root_mean_squared_error

    epsilon = 0.01
    test_size_constant = 50
    val_size_constant = 200
    template_length_constant = 5
    template_spread_constant = 5
    ts_size = 100000
    dt = 0.01
    divisor = int(0.1 / dt)
    rmses = []
    r_values = [28]
    list_ts = []
    for i, r in enumerate(r_values):
        ts = TimeSeries("Lorentz", size=ts_size, r=r, dt=dt, divisor=divisor)
        list_ts.append(ts)


    for test_size_constant in tqdm(range(1,50)):
        tsproc = TSProcessor(list_ts, template_length=template_length_constant, max_template_spread=template_spread_constant,
                         train_size=len(list_ts[0].values) - val_size_constant - test_size_constant,
                         val_size=val_size_constant, test_size=test_size_constant)
        fort, values = tsproc.pull(epsilon)
        real_values = list_ts[0].values[-test_size_constant:]
        pred_values1 = values[-test_size_constant:]
        rmses.append(root_mean_squared_error(real_values, pred_values1))
    plt.plot(rmses, label='RMSE при одном ряде',color='green')

    r_values = [28, 24.74, 30, 40, 35]
    list_ts = []

    for i, r in enumerate(r_values):
        ts = TimeSeries("Lorentz", size=ts_size, r=r, dt=dt, divisor=divisor)
        list_ts.append(ts)

    rmses2 = []

    plt.plot(rmses2, label='RMSE при нескольких рядах')
    plt.xlabel('Количество предсказываемых точек')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()