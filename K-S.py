cumulative = np.cumsum(hist)*dx               #original dataset
cumulativeHist = np.cumsum(hist1)*dx          #histogram generated
cumulativeKDE_Silverman = np.cumsum(hist2)*dx #KDE Silverman's h generated
cumulativeKDE_LSCV = np.cumsum(hist3)*dx      #KDE LSCV generated

DHist=np.max(np.absolute(cumulative-cumulativeHist))
DKDE_Silverman=np.max(np.absolute(cumulative-cumulativeKDE_Silverman))
DKDE_LSCV=np.max(np.absolute(cumulative-cumulativeKDE_LSCV))

fig, ax = plt.subplots(1,3, figsize=(16,4), sharey=True)
ax[0].set_ylim([0,1.4])
ax[0].plot(cumulative, label="Original data")
ax[0].plot(cumulativeHist, label="Histogram generated")
ax[0].legend()
ax[0].set_title('Dmax = %.3f' % DHist, y=0.05, x=0.7)

ax[1].plot(cumulative, label="Original data")
ax[1].plot(cumulativeKDE_Silverman, label="KDE_Silverman generated")
ax[1].legend()
ax[1].set_title('Dmax = %.3f' % DKDE_Silverman, y=0.05, x=0.7)

ax[2].plot(cumulative, label="Original data")
ax[2].plot(cumulativeKDE_LSCV, label="KDE_LSCV generated")
ax[2].legend()
ax[2].set_title('Dmax = %.3f' % DKDE_LSCV, y=0.05, x=0.7)

fig.savefig("Cummulative_sum_test.jpg")

lenGenerated=1000
for D in (DHist,DKDE_Silverman,DKDE_LSCV):
    if D>1.36*np.sqrt((lenDataset+lenGenerated)/(1.0*lenDataset*lenGenerated)):
        print ("Null hypothesis rejected")
    else:
        print ("Same distribution")