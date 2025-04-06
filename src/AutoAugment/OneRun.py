
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # go back one directory to src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'AutoTrain'))) # go to train directory from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) # go to train directory from src

import MonkeyPatches as MonkeyPatches # Do not delete this.
from Logger import Logger
from AutoTrain.YoloAutoTrain import YoloAutoTrainer
import pickle
import RandAugmentGenerator as RandAugmentGenerator
import torch
from multiprocessing.connection import Client
import mlflow

if __name__ == '__main__':
    serialized_data = sys.stdin.buffer.read()
    deserialized_list = pickle.loads(serialized_data)
    self = deserialized_list[0]
    n = deserialized_list[1]
    m = deserialized_list[2]
    address = deserialized_list[3]


    mlflow.set_tracking_uri(mlflow.get_tracking_uri().replace("mlruns", "runs/mlflow"))
    mlflow.set_experiment(self.augmentFinderConfig["General"]["name"])

    RandAugmentGenerator.ApplyRandAugmentMagnitude(self.default_augmentation,self.augmentFinderConfig, m)
    RandAugmentGenerator.globalN = n

    if not os.path.isfile(self.trainConfigPath): 
        Logger.LogWarning(f"Config file not found: {self.trainConfigPath}.")
        assert(False)

    Logger.LogInfo(f"Auto Train started for {self.trainConfigPath}.")

    # Prepare model and update run name
    yoloModel = YoloAutoTrainer(self.trainConfigPath)
    yoloModel.trainConfig["epochs"] = self.epochs
    yoloModel.config["train"]["train_run_name"] = f"{self.augmentFinderConfig['General']['name']}_{n}n_{m}m"
    modelSavePaths = [path.strip() for path in yoloModel.config["train"]["modelStoragePath"].split(",")]
    saveDir = modelSavePaths[0][:modelSavePaths[0].find("/")+1]
    saveName = os.path.join(saveDir, yoloModel.config['train']['train_run_name'])# + f"_n{n}_m{m}"
    yoloModel.ptSave = f"{saveName}.pt"
    yoloModel.onnxSave = f"{saveName}.onnx"

    # Train model

    try:
        trainResult = yoloModel.TrainModel()
        metricResult = trainResult.results_dict[self.metricType]
    except Exception as e:
        Logger.LogError(f"Error in training: {e}")
        metricResult = -1
    # Send the result back to the parent process
    conn = Client(address)
    conn.send(metricResult)
    conn.close()

    del yoloModel
    torch.cuda.empty_cache()