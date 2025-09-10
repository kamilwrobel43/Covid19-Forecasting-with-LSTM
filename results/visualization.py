import matplotlib
import matplotlib.pyplot as plt
def plot_predictions(preds, labels, dates):
    dates = matplotlib.dates.date2num(dates)
    plt.plot_date(dates, preds, "-", label="predicted")
    plt.plot_date(dates, labels, "-", label="real")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()