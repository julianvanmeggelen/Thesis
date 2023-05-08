import sys
sys.path.append('../')
import json
import os
import torch
import pickle
from datetime import datetime
from BasicAutoEncoder.model import AutoEncoder
import ml_collections as mlc

RESULT_DIR = "./Results/"

def logResults(trainHist: dict, model: AutoEncoder, cfg: mlc.ConfigDict):
    dgpIndex = cfg.saved_index 
    if not os.path.exists(f"{RESULT_DIR}{dgpIndex}"):
        os.mkdir(f"{RESULT_DIR}{dgpIndex}")

    i = len(os.listdir(f"{RESULT_DIR}{dgpIndex}"))
    dt = datetime.now()
    dir =  f"{dgpIndex}/{i}_{dt.strftime('%m%d%Y_%H:%M:%S')}"
    print(f"{RESULT_DIR}{dir}")
    os.mkdir(f"{RESULT_DIR}{dir}")

    #save trainhist
    with open(f"{RESULT_DIR}{dir}/trainHist.json", 'w') as f:
        json.dump(trainHist, f)

    #save model
    torch.save(model, f"{RESULT_DIR}{dir}/autoEncoder.pt")

    #save config
    with open(f"{RESULT_DIR}{dir}/cfg.pkl", 'wb') as f:
        pickle.dump(cfg, f)
    

def loadResults(dgpIndex, name):
    if not os.path.exists(f"{RESULT_DIR}{dgpIndex}"):
        raise ValueError("Unknown dgpIndex {dgpIndex}")
    
    dir =  f"{dgpIndex}/{name}"

    #load trainhist
    with open(f"{RESULT_DIR}{dir}/trainHist.json", 'r') as f:
        trainHist = json.load(f)

    #save model
    model = torch.load(f"{RESULT_DIR}{dir}/autoEncoder.pt")

    #save config
    with open(f"{RESULT_DIR}{dir}/cfg.pkl", 'rb') as f:
        cfg = pickle.load(f)

    return trainHist, model, cfg

    




