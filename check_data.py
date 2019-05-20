import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    dat = pd.read_csv("./data.csv")
    print(dat[["acceptance", "energytotal"]])
    fig, ax = plt.subplots()
    ax.plot(dat["energytotal"])
    ax.plot([0, 200], [-14.342948927294387,-14.342948927294387])
    plt.show()
