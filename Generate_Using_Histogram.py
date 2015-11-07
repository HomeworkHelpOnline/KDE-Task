import numpy as np
import matplotlib.pyplot as plt

#import data
#conists of one column of datapoints as 2.231, -0.1516, 1.564, etc
data=np.loadtxt("dataset1.txt")

#normalized histogram of loaded datase
hist, bins = np.histogram(data,bins=100,range=(np.min(data),np.max(data)) ,density=True)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2


#generate data with double random()
generatedData=np.zeros(1000)
maxData=np.max(data)
minData=np.min(data)
i=0
while i<1000:
    randNo=np.random.rand(1)*(maxData-minData)-np.absolute(minData)
    if np.random.rand(1)<=hist[np.argmax(randNo<(center+(bins[1] - bins[0])/2))-1]:
        generatedData[i]=randNo
        i+=1

#normalized histogram of generatedData
hist1, bins1 = np.histogram(generatedData,bins=100,range=(np.min(data),np.max(data)), density=True)
width2 = 0.7 * (bins1[1] - bins1[0])
center2 = (bins1[:-1] + bins1[1:]) / 2

#plot both histograms
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(14, 4), sharex=True, sharey=True)
ax1.set_title('Original data')
ax1.bar(center, hist, align='center', width=width2, fc='#AAAAFF')
ax2.set_title('Generated data')
ax2.bar(center2, hist1, align='center', width=width2, fc='#AAAAFF')

fig.savefig("Loaded_and_Generated_Data.jpg")