import csv
import math
import itertools

directory = '/root/share/NUIW_Simulations/LakeDrainage/'

def dialogue():
    paramRangeDict = dict()
    parameterList = []
    
    # ask for mesh
    meshList = input("Please enter the Mesh you want to study on (if you want to study multiple meshes just enter all of them separated by a space):\t").split()
    while not('y' == input(f'You want to study on these meshes: {meshList} \nIs that correct (y/n)?\t')):
        meshList = input("Please enter the Mesh you want to study on (if you want to study multiple meshes just enter all of them separated by a space):\t").split()
    paramRangeDict['meshInputFile'] = meshList
    

    # ask for bcType
    bcList = input("Please enter all boundary condition types you want to study (separated by a space)\t").split()
    bcList = [int(x) for x in bcList]
    while not('y' == input(f'You want to study bcTypes: {bcList} \nIs that correct (y/n)?\t')):
        bcList = input("Please enter all boundary condition types you want to study (separated by a space)\t").split()
        bcList = [int(x) for x in bcList]
    paramRangeDict['bcType'] = bcList

    # ask for parameters to study
    parameterList = input("Please enter all parameters you want to study (separated by a space)\t").split()
    while not('y' == input(f'You want to study these parameters: {parameterList} \nIs that correct (y/n)?\t')):
        parameterList = input("Please enter all parameters you want to study (separated by a space)\t").split()

    # ask for parameter range and step width for all 
    for parameter in parameterList:
        while(1):
            # ask for bounds
            while(1):
                bounds = input(f'Please enter the lower and upper bound of {parameter}\t').split()
                boundsEqual = False
                try:
                    lowerBound = float(bounds[0])
                    upperBound = float(bounds[1])
                    print(f"lower: {lowerBound}; upper: {upperBound}")
                    if (lowerBound == upperBound):
                        boundsEqual = True
                        stepList = [lowerBound]
                        paramRangeDict[parameter] = stepList
                    break
                except:
                    print(f"Invalid input: {bounds}")

            # ask for stepping method
            while(not(boundsEqual)):
                stepIn = input(f'Please enter how the parameters should be stepped through (or help for more info)\t').split()
                try:                    
                    match stepIn[0]:
                        case 'help':
                            print('There are four ways on how to step through the parameters:\n'
                            '"+ x" adds x to lower bound until upper bound is reached\n'
                            '"* x" multiplies lower bound by x until upper bound is reached\n'
                            '"step x" creates x equally spaced parameters ')
                        case "+":
                            stepSize = float(stepIn[1])
                            stepList = [lowerBound + i * stepSize for i in range(int((upperBound - lowerBound) // stepSize + 1))]
                            if stepList[-1] != upperBound:
                                stepList.append(upperBound)
                            paramRangeDict[parameter] = stepList
                            break
                        case "*":
                            stepSize = float(stepIn[1])
                            stepList = [lowerBound * (stepSize**i) for i in range(int(math.log(upperBound / lowerBound, stepSize))+ 1)]
                            if stepList[-1] != upperBound:
                                stepList.append(upperBound)
                            paramRangeDict[parameter] = stepList
                            break
                        case "step":
                            stepNumber = float(stepIn[1]) - 1
                            stepSize = (upperBound - lowerBound) / stepNumber
                            stepList = [lowerBound + i * stepSize for i in range(int((upperBound - lowerBound) // stepSize + 1))]
                            if stepList[-1] != upperBound:
                                stepList.append(upperBound)
                            paramRangeDict[parameter] = stepList
                            break
                        case _:
                            print(f"Unknown input: {stepIn}")

                except:
                    print(f"Invalid input: {stepIn}")

            # ask if it went as planned
            if ('y' == input(f'For {parameter} you chose these values to study: {stepList} \nIs that as intended? (y/n)\t')):
                break
    simulationNum = 1
    for list in paramRangeDict.values():
        simulationNum = simulationNum * len(list)
    print(f"\n---This will generate {simulationNum} simulations---")
    return paramRangeDict
        
def createCSV(paramRangeDict):
    while(1):
        numFiles = input(f'Please enter how many files shall be created\t')
        try:
            numFiles = int(numFiles)
            break
        except:
            print(f"Invalid input: {numFiles}")

    # Generate all permutations of values
    values = list(paramRangeDict.values())
    keys = list(paramRangeDict.keys())
    rows = list(itertools.product(*values))

    rowsPerFile = math.ceil(len(rows) / numFiles)

    for i in range(numFiles):
        startIdx = i * rowsPerFile
        endIdx = min(startIdx + rowsPerFile, len(rows))
        csvOutFilename = f'{directory}{i+1:02}_Testsuite.csv'
        with open(csvOutFilename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                writer.writerow(keys)
                writer.writerows(rows[startIdx:endIdx])

    

paramRangeDict = dialogue()
createCSV(paramRangeDict)       
