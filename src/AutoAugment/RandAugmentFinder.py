''' 
RandAugmentFinder contains the RandAugmentFinder class, 
which is used to find the best RandAugment policy for a given dataset.

It contains the grid search algorithm.
It also contains the randaugment function that will be implemented in each cycle of the search agorithm.
'''
from Logger import Logger
import configparser
import os
import traceback
import RandAugmentGenerator as RandAugmentGenerator

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import subprocess
from multiprocessing.connection import Listener


METRIC_MAP = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'fitness']

class RandAugmentFinder:
    globalAugmentationList = []
    globalN = 0

    class Timer:
        def __init__(self, mtx):
            self.start = time.time()
            self.last = self.start
            self.runRemaining = len(mtx) * len(mtx[0])
            self.dtList = []

        def RecordAndPrintTime(self):
            # Skip starting run
            if len(self.dtList) == 0: 
                Logger.LogCustom("RandAugment", "Starting timer for auto augmentation process.")
                return

            # Record time
            dt = time.time() - self.last
            self.last = time.time()
            self.dtList.append(dt)
            self.runRemaining -= 1
            predictedTimeRemaining = np.mean(self.dtList) * self.runRemaining

            # Printing time
            elapsed_time = time.time() - self.start
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            Logger.LogCustom("RandAugment", f"Time elapsed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

            hours, rem = divmod(predictedTimeRemaining, 3600)
            minutes, seconds = divmod(rem, 60)
            Logger.LogCustom("RandAugment", f"Time remaining: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    def __init__(self, augmentFinderConfig):
        """
        Initialize the RandAugmentFinder class.
        Parameters:
        augmentFinderConfig (str / configparser.ConfigParser): The configuration file path or the configparser object.
        trainConfig (str): The yolo train configuration file path, it's a text file.
        num_epochs (int): Number of epochs per cycle.
        operation_policies (tuple(int,int,int)): Tuple of start, end and step values for the number of operations in a policy.
        magnitude_policies (tuple(float,float,float)): Tuple of start, end and step values for the magnitude of operations in a policy.
        save_dir (str): Directory to save the images with RandAugment applied.
        """
        
        self.augmentFinderConfig = self.InitStrConfig(augmentFinderConfig) if type(augmentFinderConfig) == str else augmentFinderConfig
        self.default_augmentation = self.InitStrConfig(self.augmentFinderConfig["General"]["default_augmentation"])
        self.trainConfigPath = self.augmentFinderConfig["General"]["train_config"]
        self.N_min = float(self.augmentFinderConfig["General"]["N_min"])
        self.N_max = float(self.augmentFinderConfig["General"]["N_max"])
        self.N_step = float(self.augmentFinderConfig["General"]["N_step"])
        self.M_min = float(self.augmentFinderConfig["General"]["M_min"])
        self.M_max = float(self.augmentFinderConfig["General"]["M_max"])
        self.M_step = float(self.augmentFinderConfig["General"]["M_step"])

        self.epochs = int(self.augmentFinderConfig["General"]["epochs"])
        self.debug_save_dir = self.augmentFinderConfig["General"]["debug_save_dir"]

        global METRIC_MAP
        self.metricType = METRIC_MAP[int(self.augmentFinderConfig["General"]["metric"])]
        


    def InitStrConfig(self, configPath:str):
        Logger.LogCustom("augmenter", f"Parsing augmentation config, configPath: {configPath}") 
        config = configparser.ConfigParser(inline_comment_prefixes="#")
        config.optionxform = str
        config.read(configPath)
        return config

    def GridSearch(self):
        """
        Perform grid search to find the best RandAugment policy.
        """

        # Initialize the augmentations to default nothing first.
        Logger.LogCustom("RandAugment", "Running Grid Search on randAugment.")

        try:
            nSteps = round((self.N_max - self.N_min) / self.N_step) + 1
            mSteps = round((self.M_max - self.M_min) / self.M_step) + 1

            metricTrackingMatrix = [ [0 for _ in range(mSteps)]  for _ in range(nSteps) ] # arr[n][m]
            timer = RandAugmentFinder.Timer(metricTrackingMatrix)
            import tracemalloc

            tracemalloc.start()

            for _n in range(nSteps):
                for _m in range(mSteps):
                    timer.RecordAndPrintTime()
                    n = int(self.N_min + _n * self.N_step)
                    m = RandAugmentFinder.RemoveLeadingZerosAndNine(self.M_min + _m * self.M_step)
                    # Initialize randaugment magnitude and num of operations
                    Logger.LogCustom("RandAugment", f"Running RandAugment with N = {n}, M = {m}")
                    RandAugmentGenerator.ApplyRandAugmentMagnitude(self.default_augmentation,self.augmentFinderConfig, m)
                    RandAugmentGenerator.globalN = n

                    # TODO: Run training here and get the result. 
                    # Perform test on the validation set and get the result.
                    # Save the result in a grid search table
                    try:
                        if not os.path.isfile(self.trainConfigPath): 
                            Logger.LogWarning(f"Config file not found: {self.trainConfigPath}.")
                            continue

                        Logger.LogInfo(f"Auto Train started for {self.trainConfigPath}.")
                        
						# Create a listener for communication
                        address = ('localhost', 6002)
                        listener = Listener(address)
                        
                        data = pickle.dumps([self, n, m, address])
                        process = subprocess.Popen(['python', "src/AutoAugment/OneRun.py"], stdin=subprocess.PIPE)
                        
                        stdout, stderr = process.communicate(input=data)
                        conn = listener.accept()
                        metricResult = conn.recv()
                        conn.close()
                        listener.close()
                        
                        metricTrackingMatrix[_n][_m] = metricResult

                        process.wait()

                    except Exception as e:
                        Logger.LogError(f"Error running Auto Train for {self.trainConfigPath}: {traceback.format_exc()}")

            self.SaveMetricTrackingMatrix(metricTrackingMatrix)
        except:
            Logger.LogError(traceback.format_exc())
    
    Logger.LogCustom("RandAugment", "Completed Grid Search on randAugment.")


    # Prepare the N and M first.
    # Prepare the parameters first, the magnitude, store all transformation in a list.
    # To use it in the monkey patching, there is no need to set the probability
    # Just set the magnitude of the transformation, because later we are going to choose the number of augmentation randomly to apply. 

    def RemoveLeadingZerosAndNine(num, tolerance: int = 5):
        ''' Removes leading zeros from float numbers. Turns something like 12.30000005 into 12.3 '''
        # Remove leading zeros from the number
        if type(num) != float:
            raise ValueError(f"RemoveLeadingZeros only works with float, value inserted was: {num}")
        num = str(num)
        begin = False
        trackPos = 0
        counter = 0
        for i in range(len(num)):
            if num[i] == '.':
                begin = True
                trackPos = i
                continue
            if not begin: continue
            if num[i] != '0':
                trackPos = i
                counter = 0
            else:
                counter += 1
                if counter >= tolerance:
                    return float(num[:trackPos+1])
        
        # Remove trailing nines from the number
        
        # Find the position of '.' in the number
        pos_dot = num.find('.')
        if pos_dot != -1:
            # Find the position of '999' after the '.'
            val9 = '9'*tolerance
            pos_999 = num.find(val9, pos_dot)
            if pos_999 != -1:
                num = num[:pos_999]
                if num[-1] == '.': num = num[:-1]
                num = num[:-1] + str(int(num[-1])+1)
        
        return float(num)


    def SaveMetricTrackingMatrix(self, matrix):

        Logger.LogCustom("RandAugment", "Saving Metric Tracking Matrix")
        # Create a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, cmap="Blues")

        nSteps = round((self.N_max - self.N_min) / self.N_step) + 1
        mSteps = round((self.M_max - self.M_min) / self.M_step) + 1
        nLabels = [int(self.N_min + _n * self.N_step) for _n in range(nSteps)]
        mLabels = [RandAugmentFinder.RemoveLeadingZerosAndNine(self.M_min + _m * self.M_step) for _m in range(mSteps)]
        
        plt.xticks(ticks=np.arange(len(matrix[0])) + 0.5, labels=mLabels, fontsize=12)
        plt.yticks(ticks=np.arange(len(matrix)) + 0.5, labels=nLabels, fontsize=12)
        plt.title("Grid Search Results Heatmap")
        plt.xlabel("M values")
        plt.ylabel("N values")
        # Ensure the directory exists
        os.makedirs(self.debug_save_dir, exist_ok=True)
        plt.savefig(os.path.join(self.debug_save_dir, self.augmentFinderConfig["General"]["name"] + ".png"))

        # Find the best parameters
        best_indices = sorted(((i, j) for i in range(len(matrix)) for j in range(len(matrix[i]))), key=lambda x: matrix[x[0]][x[1]], reverse=True)[:3]
        for idx, (best_n, best_m) in enumerate(best_indices):
            best_metric = matrix[best_n][best_m]
            n = nLabels[best_n]
            m = mLabels[best_m]
            Logger.LogCustom("RandAugment", f"Top {idx + 1} - N: {n}, M: {m}, Metric: {best_metric}")
    





