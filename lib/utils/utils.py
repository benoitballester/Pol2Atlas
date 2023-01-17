import subprocess
import pickle
import os

def createDir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        print(f"Directory {path} already exists !")

def runScript(script, argumentList, outFile=None):
    # Runs the command as a standard bash command
    # script is command name without path
    # argumentList is the list of argument that will be passed to the command
    if outFile == None:
        subprocess.run([script] + argumentList)
    else:
        with open(outFile, "wb") as outdir:
            subprocess.run([script] + argumentList, stdout=outdir)


def dump(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)