import csv
import time

class CSVSaver:    
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename,"w")
        self.csv_writer = csv.writer(self.file)
        
    def save_data_item(self, data):
        self.csv_writer.writerow(data)

def plot_csv_data(filename):
    import matplotlib.pyplot as plt
    import numpy as np

    titles = ['x','y','z']
    f = open(filename,"r")
    reader = csv.reader(f)

    str_list = list(reader)
    print str_list[0]
    float_list = [[float(i[0]),float(i[1]),float(i[2]),float(i[3])] for i in str_list]

    data = np.array(float_list).T
    start_time = np.min(data[3])

    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(data[3] - start_time,data[i])
        plt.title(titles[i])

    plt.show()