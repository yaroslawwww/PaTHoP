from TimeSeries import TimeSeries

def main():
    row = TimeSeries("Lorentz", 10000)
    row.split_train_val_test()
    row.print(500)


if __name__ == '__main__':
    main()