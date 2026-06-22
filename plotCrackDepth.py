import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re

# function for neater plotting later
def plotGlacier(plotNum, plotFilename, stressArr, crackDepthArr, label):
    plt.figure(plotNum)
    plt.plot(stressArr, crackDepthArr, label=label)
    plt.title('Crack Depth')
    plt.xlabel("$\sigma_{xx}$ in $N/m^2$ top left")
    plt.ylabel('Crack Depth in m')
    plt.legend()
    plt.savefig(plotFilename, dpi=300)


mainDirectory = '/root/share/NUIW_Simulations/LakeDrainage/results/'
subFolder = "00_SimpleTest"

# get complete path of npy arrays in folders
rawStressFileList = glob(f"{mainDirectory}{subFolder}/*stress.npy")
rawCrackDepthFileList = glob(f"{mainDirectory}{subFolder}/*crackDepth.npy")

# extract the filename of the arrays from the complete path
stressFileList = []
crackDepthList = []
for path in rawStressFileList:
    stressFileList.append(re.search("[0-9]*_[a-zA-Z0-9_]*.npy",path).group())

for path in rawCrackDepthFileList:
    crackDepthList.append(re.search("[0-9]*_[a-zA-Z0-9_]*.npy",path).group())

# plot crack depth making sure that the right arrays are used by comparing the numbers in their respective filename
for i, stressFileName in enumerate(stressFileList):
    stressPathNum = re.search("[0-9]*", stressFileName).group()
    for crackFileName in crackDepthList:
        crackPathNum = re.search("[0-9]*",crackFileName).group()

        if crackPathNum == stressPathNum:
            stressArr = np.load(f"{mainDirectory}{subFolder}/{stressFileName}")
            crackDepthArr = np.load(f"{mainDirectory}{subFolder}/{crackFileName}")
            plotGlacier(
                plotNum=i+1,
                plotFilename=f"{mainDirectory}{subFolder}/{stressPathNum}_CrackDepthPlot",
                stressArr=stressArr,
                crackDepthArr=crackDepthArr,
                label=f'Sim num: {stressPathNum}'
            )
            plotGlacier(
                plotNum=0,
                plotFilename=f"{mainDirectory}{subFolder}/00_CrackDepthGeneralPlot",
                stressArr=stressArr,
                crackDepthArr=crackDepthArr,
                label=f'Sim num: {stressPathNum}'
            )
            continue   