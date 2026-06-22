from source import glacierSimulator as sim
from source import helperFunc as hf
from source.validKeys import validKeysTime, validKeysBoundary, validKeysGlacier
from csv import DictReader, DictWriter
from datetime import datetime, timezone
import argparse
import os
from pathlib import Path

from mpi4py import MPI

# initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# adjust for different file system if necessary
CODEDIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTDIR = CODEDIR / "results"
MESHDIR = CODEDIR / "mesh"

def runSimulation(filename, meshName, simulationValues):
    """Creates all classes necessary for simulation and runs that simulation using simulation values

    Parameters
    ----------
    filename : string
        name of output file
    meshName : string
        name of input mesh
    simulationValues : dictionary
        parameter values associated to valid keys necessary for simulation

    Returns
    -------
    glacierSimulator
        class instance after simulation was run
    """
    reg = sim.Region(meshInputFile=meshName)
    boundary = sim.Boundary(
        mesh=reg,
        **hf.filterDict(simulationValues, validKeysBoundary)
    )
    boundary.defineStrongBCs()
    timeControl = sim.TimeStepControl(**hf.filterDict(simulationValues, validKeysTime))
    glacier = sim.GlacierSimulator(
        filename=filename,
        mesh=reg,bc=boundary,
        timeControl=timeControl,
        **hf.filterDict(simulationValues, validKeysGlacier)
    )
    glacier.simulate()
    return glacier

def storeMetadata(csvOutFilename, simulationValuesDicts):
    """stores Dictionary data to csv

    Parameters
    ----------
    csvOutFilename : string
        output csv filename
    simulationValuesDicts : array of dicts
        dicts of used simulation values
    """
    with open(csvOutFilename, 'w', newline='') as csvfile:
        writer = DictWriter(csvfile, fieldnames=simulationValuesDicts[0].keys())
        writer.writeheader()
        for row in simulationValuesDicts:
            writer.writerow(row)
    

def main():
    """reads input data from csv, simulates and saves results (also metadata and crack depth info)
    """
    parser = argparse.ArgumentParser(description="CSV-Datei")
    parser.add_argument('job_name')
    parser.add_argument('csv_file')
    args = parser.parse_args()
    
    subfolderName = f'{args.job_name}_sim'
    csvInFilename = f'{CODEDIR}/{args.csv_file}'
    csvOutFilename = f'{RESULTDIR}/{subfolderName}/{args.job_name}_Metadata.csv'
    simulationValues = []
    glaciers = []
    metadataDictList = []

    # generate dict from csv 
    with open(csvInFilename, newline='') as csvfile:
        reader = DictReader(csvfile,delimiter=';')
        for row in reader:
            simulationValues.append(row)

    # simulating
    for i, values in enumerate(simulationValues):
        if rank == 0:
            print(f"\n--- Simulation {i+1}/{len(simulationValues)} ---")

        # save time and filename of simulation for metadata
        metadataDict = dict()
        metadataDictList.append(metadataDict)
        timeUTC = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        simFilename = f"{RESULTDIR}/{subfolderName}/{i+1:02}_{args.job_name}_out"
        metadataDictList[i]['datetimeUTC'] = timeUTC
        metadataDictList[i]['filename'] = simFilename
        metadataDictList[i]['meshInputFile'] = values['meshInputFile']

        # run simulation and save simulating unit (glacier) in list
        glaciers.append(runSimulation(
            filename=simFilename,
            meshName=f'{MESHDIR}/{values["meshInputFile"]}',
            simulationValues=values,
            )
        )
        glaciers[i].saveCrackDepth()
    # saving crackDepth, stress and meta- Data
    for i, glacier in enumerate(glaciers):
        metadataDictList[i] = {**metadataDictList[i], **glacier.getInstanceVariables()}

    storeMetadata(csvOutFilename,metadataDictList)
    

# execute programm
if __name__ == "__main__":
    main()