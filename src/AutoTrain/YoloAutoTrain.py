''' This auto trainer takes in a config file and trains the model based on the config file. '''
import configparser
import os
from Logger import Logger
from ultralytics import YOLO, settings
import torch
import mlflow
import cv2
from AutoTrainSetup import ToggleAutoAugmentation, autoaugment
import shutil
'''
starterConfig = configparser.ConfigParser(inline_comment_prefixes="#")
starterConfig.optionxform = str
starterConfig.read("asset/config.conf")
'''

class YoloAutoTrainer:

	def __init__(self, configPath: str):
		Logger.LogInfo("Initializing YoloAutoTrain.")
		self.LatestMetrics = None
		self.ptSave = None
		self.onnxSave = None

		self.configPath = configPath
		self.PrepConfig()
		self.MakeModel()
		self.PrepTrainConfig()
		settings.update({"mlflow": bool(int(self.config["log"]["mlflow"]))})\
	
	def __del__(self):
		Logger.LogInfo("Deconstructing Yolo")
		del self.model
		del self.config
		del self.trainConfig
		del self.LatestMetrics
		del self.ptSave
		del self.onnxSave
		del self.imgsz
		del self.trainResult
		Logger.LogInfo("Deconstruction completed.")

	def PrepConfig(self):
		Logger.printConsole = True
		self.config = configparser.ConfigParser(inline_comment_prefixes="#")
		self.config.optionxform = str
		self.config.read(self.configPath)

		# if there self.config["mode"]["train"] and self.config["mode"]["train"] are both true or false, then it is invalid.
		if bool(int(self.config["mode"]["train"])) == bool(int(self.config["mode"]["test"])): 
			raise Exception("Invalid config file, both train and test mode are active or inactive.")

	def MakeModel(self):
		# Make a model based on the config file
		Logger.LogInfo("Making model based on the config file.")
		self.model = YOLO(model=self.config["model"]["modelpath"])

		# Initial predict, this is required to start up the YOLO8. If not, it will lag later on the first inference.
		self.imgsz = (int(self.config["train"]["imgszX"]), int(self.config["train"]["imgszY"]))
		
		try:
			self.model.predict(cv2.imread(self.config["model"]["nullImgPath"]), imgsz=self.imgsz)
			Logger.LogInfo(f"Imgsz not changed, {self.imgsz}")
		except Exception as e:
			errorLines = str(e).split("\n")
			imgsz1 = int(errorLines[-3].split(":")[-1].strip())
			imgsz2 = int(errorLines[-2].split(":")[-1].strip())
			self.imgsz = (imgsz1, imgsz2)
			Logger.LogWarning(f"Incorrect Imgsz detected, setting imgsz to {self.imgsz}")

	def PrepTrainConfig(self):
		# Prepare the training config based on the config file
		Logger.LogInfo("Preparing training config based on the config file.")

		self.trainConfig = {
			# "model": str(self.config["train"]["model"]),
			"data": str(self.config["train"]["data"]),
			"epochs": int(self.config["train"]["epochs"]),
			"patience": int(self.config["train"]["patience"]),
			"batch": int(self.config["train"]["batch"]),
			"imgsz": (int(self.config["train"]["imgszX"]), int(self.config["train"]["imgszY"])),
			"save": bool(int(self.config["train"]["save"])),
			"save_period": int(self.config["train"]["save_period"]),
			"cache": bool(int(self.config["train"]["cache"])),
			"device": "cuda:0" if torch.cuda.is_available() else "cpu",
			"workers": int(self.config["train"]["workers"]),
			"project": str(self.config["train"]["project"]),
			"name": str(self.config["train"]["name"]),
			"exist_ok": bool(int(self.config["train"]["exist_ok"])),
			"pretrained": bool(int(self.config["train"]["pretrained"])),
			"optimizer": str(self.config["train"]["optimizer"]),
			"verbose": bool(int(self.config["train"]["verbose"])),
			"seed": int(self.config["train"]["seed"]),
			"deterministic": bool(int(self.config["train"]["deterministic"])),
			"single_cls": bool(int(self.config["train"]["single_cls"])),
			"rect": bool(int(self.config["train"]["rect"])),
			"cos_lr": bool(int(self.config["train"]["cos_lr"])),
			"close_mosaic": int(self.config["train"]["close_mosaic"]),
			"resume": bool(int(self.config["train"]["resume"])),
			"amp": bool(int(self.config["train"]["amp"])),
			"fraction": float(self.config["train"]["fraction"]),
			"profile": bool(int(self.config["train"]["profile"])),
			"freeze": int(self.config["train"]["freeze"]),
			"lr0": float(self.config["train"]["lr0"]),
			"lrf": float(self.config["train"]["lrf"]),
			"momentum": float(self.config["train"]["momentum"]),
			"weight_decay": float(self.config["train"]["weight_decay"]),
			"warmup_epochs": float(self.config["train"]["warmup_epochs"]),
			"warmup_momentum": float(self.config["train"]["warmup_momentum"]),
			"warmup_bias_lr": float(self.config["train"]["warmup_bias_lr"]),
			"box": float(self.config["train"]["box"]),
			"cls": float(self.config["train"]["cls"]),
			"dfl": float(self.config["train"]["dfl"]),
			"pose": float(self.config["train"]["pose"]),
			"kobj": float(self.config["train"]["kobj"]),
			"label_smoothing": float(self.config["train"]["label_smoothing"]),
			"nbs": int(self.config["train"]["nbs"]),
			"mask_ratio": int(self.config["train"]["mask_ratio"]),
			"dropout": float(self.config["train"]["dropout"]),
			"val": bool(int(self.config["train"]["val"])),
			"plots": bool(int(self.config["train"]["plots"])),
			# "augment": False,
        	# "hsv_h": 0.0,
        	# "hsv_s": 0.0,
        	# "hsv_v": 0.0,
        	# "degrees": 0.0,
        	# "translate": 0.0,
        	# "scale": 0.0,
        	# "shear": 0.0,
        	# "perspective": 0.0,
        	# "flipud": 0.0,
        	# "fliplr": 0.0,
        	# "bgr": 0.0,
        	# "mosaic": 0.0,
        	# "copy_paste": 0.0,
        	# "auto_augment": "randaugment",
		}

		for key in self.trainConfig:
			if isinstance(self.trainConfig[key], str) and self.trainConfig[key] == "None":
				self.trainConfig[key] = None

		# saving model config
		modelSavePaths = [path.strip() for path in self.config["train"]["modelStoragePath"].split(",")]
		for path in modelSavePaths:
			if ".pt" in path: self.ptSave = path
			elif ".onnx" in path: self.onnxSave = path
			else: continue
			if not os.path.exists(os.path.dirname(path)): os.makedirs(os.path.dirname(path))
		

		try:
			Logger.LogInfo("Setting up auto augmentation status")
			auto_augmentation = bool(int(self.config["train"]["auto_augmentation"]))
			global autoaugment
			if auto_augmentation: ToggleAutoAugmentation(auto_augmentation)
			else: autoaugment = False
		except Exception as e:
			Logger.LogError(f"Error setting auto augmentation: {e}")

	def TrainModel(self):
		# Train the model based on the config file and export as onnx file
		if not bool(int(self.config["mode"]["train"])): return

		Logger.LogInfo("Training model based on the config file.")

		if bool(int(self.config["log"]["mlflow"])):
			with mlflow.start_run(run_name=self.config["train"]["train_run_name"]):
				self.trainResult = self.model.train(**self.trainConfig)
				mlflow.end_run()
		else:
			self.trainResult = self.model.train(**self.trainConfig)

		if self.ptSave is not None: self.model.save(filename=self.ptSave)

		if bool(int(self.config["log"]["export_onnx"])): 
			path = self.model.export(format="onnx")  # export the model to ONNX format
			if self.onnxSave is not None: shutil.copy(path, self.onnxSave)
			Logger.LogInfo(f"Onnx model exported to {path}.")
		return self.trainResult
			
	
	def EvalModel(self):
		# Evaluate the model based on the config file
		if not bool(int(self.config["mode"]["test"])): return

		Logger.LogInfo("Evaluating model based on the config file.")
		self.LatestMetrics = self.model.val(data=self.config["test"]["data"], imgsz = self.imgsz, batch = int(self.config["test"]["batch"]))

		self.EvalLogMLFlow()
	
	def EvalLogMLFlow(self):
		if not bool(int(self.config["log"]["mlflow"])): return
		
		# Log the evaluation results to mlflow
		Logger.LogInfo("Logging evaluation results to mlflow.")

		saveDir = str(self.LatestMetrics.save_dir).replace("\\", "/")
		runName = saveDir.split("/")[-1]

		# try:
		# 	print("Latest Metrics keys: ", self.LatestMetrics.keys())
		# except:
		# 	pass
		# print("Latest Metrics: ", self.LatestMetrics)

		with mlflow.start_run(run_name=runName) as run:
			modelPath = os.path.join(os.getcwd(), self.config["model"]["modelpath"])
			runPath = os.path.join(os.getcwd(), str(self.LatestMetrics.save_dir)).replace("\\", "/")

			mlflow.log_param("Eval Model Name", os.path.basename(self.config["model"]["modelpath"]))
			mlflow.log_param("Eval Run Name", str(self.LatestMetrics.save_dir).split("\\")[-1])
			mlflow.log_param("Eval Model Path", modelPath)
			mlflow.log_param("Eval Data Path", self.config["test"]["data"])
			mlflow.log_param("Eval Run Path", runPath)
			mlflow.log_param("task", self.LatestMetrics.task)

			# Save the artifacts
			for file in os.listdir(saveDir):
				mlflow.log_artifact(os.path.join(saveDir, file), os.path.basename(file))
			mlflow.log_artifact(modelPath)
			mlflow.log_artifact(runPath)
			mlflow.log_artifact(self.config["test"]["data"])
			mlflow.log_artifact(self.configPath)

			# Save the metrics as a dictionary
			result_dict:dict = self.LatestMetrics.results_dict
			result_dict["fitness"] = self.LatestMetrics.fitness
			result_dict = {key.replace("(", "").replace(")", ""): value for key, value in result_dict.items()}
			# print("result_dict 1: ", result_dict)
			# result_dict = {key: value for key, value in result_dict.items() if all(c.isalnum() or c in "_-./ " for c in key)}
			# print("result_dict: ", result_dict)
			mlflow.log_metrics(result_dict)
			mlflow.log_metrics(self.LatestMetrics.speed)

