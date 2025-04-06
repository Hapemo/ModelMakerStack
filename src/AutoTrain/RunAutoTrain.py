
import os
from Logger import Logger
import torch
from YoloAutoTrain import YoloAutoTrainer
import mlflow
import json
import configparser
import yaml
import traceback

Logger.printConsole = True

def RunAutoTrain(configFileDir, configFiles, experimentName):
	# configFiles = ['MA_eval_AdvancedBlur.conf', 'MA_eval_Affine_1.2.conf', 'MA_eval_Affine_1.4.conf', 'MA_eval_Affine_1.6.conf', 'MA_eval_Affine_1.8.conf', 'MA_eval_Affine_2.conf', 'MA_eval_CLAHE.conf', 'MA_eval_CoarseDropout.conf', 'MA_eval_ColorJitter.conf', 'MA_eval_D4.conf', 'MA_eval_Emboss.conf', 'MA_eval_GaussNoise.conf', 'MA_eval_ImageCompression.conf', 'MA_eval_ISONoise.conf', 'MA_eval_MultiplicativeNoise.conf', 'MA_eval_Perspective.conf', 'MA_eval_PixelDropout_0.05_False.conf', 'MA_eval_PixelDropout_0.05_True.conf', 'MA_eval_PixelDropout_0.07_False.conf', 'MA_eval_PixelDropout_0.07_True.conf', 'MA_eval_PixelDropout_0.1_False.conf', 'MA_eval_PixelDropout_0.1_True.conf', 'MA_eval_RandomFog.conf', 'MA_eval_RandomScale_-0.82.conf', 'MA_eval_RandomScale_-0.84.conf', 'MA_eval_RandomScale_-0.86.conf', 'MA_eval_RandomScale_-0.88.conf', 'MA_eval_RandomScale_-0.9.conf', 'MA_eval_RandomShadow.conf', 'MA_eval_RingingOvershoot.conf', 'MA_eval_SafeRotate.conf', 'MA_eval_AdvancedBlur_10.conf', 'MA_eval_Affine_1.2_10.conf', 'MA_eval_Affine_1.4_10.conf', 'MA_eval_Affine_1.6_10.conf', 'MA_eval_Affine_1.8_10.conf', 'MA_eval_Affine_2_10.conf', 'MA_eval_CLAHE_10.conf', 'MA_eval_CoarseDropout_10.conf', 'MA_eval_ColorJitter_10.conf', 'MA_eval_D4_10.conf', 'MA_eval_Emboss_10.conf', 'MA_eval_GaussNoise_10.conf', 'MA_eval_ImageCompression_10.conf', 'MA_eval_ISONoise_10.conf', 'MA_eval_MultiplicativeNoise_10.conf', 'MA_eval_Perspective_10.conf', 'MA_eval_PixelDropout_0.05_False_10.conf', 'MA_eval_PixelDropout_0.05_True_10.conf', 'MA_eval_PixelDropout_0.07_False_10.conf', 'MA_eval_PixelDropout_0.07_True_10.conf', 'MA_eval_PixelDropout_0.1_False_10.conf', 'MA_eval_PixelDropout_0.1_True_10.conf', 'MA_eval_RandomFog_10.conf', 'MA_eval_RandomScale_-0.82_10.conf', 'MA_eval_RandomScale_-0.84_10.conf', 'MA_eval_RandomScale_-0.86_10.conf', 'MA_eval_RandomScale_-0.88_10.conf', 'MA_eval_RandomScale_-0.9_10.conf', 'MA_eval_RandomShadow_10.conf', 'MA_eval_RingingOvershoot_10.conf', 'MA_eval_SafeRotate_10.conf']
	Logger.printConsole = True

	mlflow.set_tracking_uri(mlflow.get_tracking_uri().replace("mlruns", "runs/mlflow"))

	Logger.LogInfo("Starting Auto Train.")
	runCount = 0
	for configFile in configFiles:
		mlflow.set_experiment(experimentName)
		try:
			configPath = os.path.join(configFileDir, configFile)
			if not os.path.isfile(configPath): 
				Logger.LogWarning(f"Config file not found: {configPath}.")
				continue
			runCount += 1
			Logger.LogInfo(f"Auto Train started for {configPath}.")
			
			# Prepare model
			yoloModel = YoloAutoTrainer(configPath)
			# Train model
			yoloModel.TrainModel()

			# Eval model
			yoloModel.EvalModel()

			# Clear memory
			del yoloModel
			torch.cuda.empty_cache()
			
		except Exception as e:
			Logger.LogError(f"Error running Auto Train for {configFile}: {traceback.format_exc()}")
	
	Logger.LogInfo(f"Auto Train completed. {runCount} runs completed.")

def VerifyFileCounts(directory):
	''' Takes in a directory and verifies the number of text files and image files in the directory. 
		Returns True if the number of text files and image files do not match or if there are no images in the directory.'''
	text_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
	image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

	text_file_count = len(text_files)
	image_file_count = len(image_files)

	if text_file_count != image_file_count:
		Logger.LogWarning(f"Mismatch between number of text files and image files in {directory}")
		return True
	if text_file_count == 0:
		Logger.LogWarning(f"No image found in {directory}")
		return True
	
	return False

def VerifyDatasetPath(ultralytics_settings_path, configFileDir, configFiles, experimentFilePath):
	''' Takes in the ultralytics setting file and the dataset path from config file, then check if the dataset path is correct.
	'''
	passed = True
	modelList = []
	# Verify ultralytics setting file directory
	Logger.LogInfo(f"Verifying dataset directory in the config file: {ultralytics_settings_path}")
	with open(ultralytics_settings_path, 'r') as f:
		config = json.load(f)
	
	ultralyticsDatasetsDir = config.get('datasets_dir')
	if not ultralyticsDatasetsDir:
		Logger.LogError(f"Ultralytics Datasets Dir not found in the config file ({ultralytics_settings_path}).")
		passed = False
	
	if not os.path.isdir(ultralyticsDatasetsDir):
		Logger.LogError(f"Dataset directory does not exist ({ultralyticsDatasetsDir}), please check the datasets_dir variable in {ultralytics_settings_path}.")
		passed = False

	# Verify config file directory
	if not os.path.isdir(configFileDir):
		Logger.LogError(f"Config file directory does not exist ({configFileDir}), please double check configFileDirectory in ({experimentFilePath}).")
		passed = False

	for file in configFiles:
		configPath = os.path.join(configFileDir, file)
		if not os.path.isfile(configPath):
			Logger.LogError(f"Config file not found: {configPath}, please double check configFiles in ({experimentFilePath}).")
			passed = False

		# Load in run config file 
		testConfig = configparser.ConfigParser(inline_comment_prefixes="#")
		testConfig.optionxform = str 
		testConfig.read(configPath)
		testMode = bool(int(testConfig["mode"]["test"]))
		trainMode = bool(int(testConfig["mode"]["train"]))

		# Verify pretrained model file path
		pretrainedModelPath = testConfig["model"]["modelpath"]
		nullImagePath = testConfig["model"]["nullImgPath"]
		if (pretrainedModelPath not in modelList) and (not os.path.isfile(pretrainedModelPath)):
			Logger.LogError(f"Pretrained model file does not exist ({pretrainedModelPath}), please check the [model][modelpath] variable in ({configPath}).")
			passed = False

		if not os.path.isfile(nullImagePath):
			Logger.LogError(f"Null image file does not exist ({nullImagePath}), please check the [model][nullImgPath] variable in ({configPath}).")
			passed = False
		
		# Store the trained model in modelList
		if trainMode:
			modelsGenerated = [ model.strip() for model in testConfig["train"]["modelStoragePath"].split(",")]
			modelList.extend(modelsGenerated)
		
		# Verify dataset yaml config path 
		datasetYamlPath = []
		if testMode:
			testDatasetPath = testConfig["test"]["data"]
			if not os.path.isfile(testDatasetPath):
				Logger.LogError(f"Test dataset directory does not exist ({testDatasetPath}), please check the [test][data] variable in ({configPath}).")
				passed = False
			else:
				datasetYamlPath.append(testDatasetPath)
		if trainMode:
			trainDatasetPath = testConfig["train"]["data"]
			if not os.path.isfile(trainDatasetPath):
				Logger.LogError(f"Train dataset directory does not exist ({trainDatasetPath}), please check the [train][data] variable in ({configPath}).")
				passed = False
			else:
				datasetYamlPath.append(trainDatasetPath)
		
		for path in datasetYamlPath:
			Logger.LogInfo(f"Verifying dataset yaml file: {path}")
			with open(path, 'r') as stream:
				try:
					data = yaml.safe_load(stream)

					dataset_root = data.get('path')
					train_images = data.get('train') or []
					val_images = data.get('val') or []

					all_images = train_images + val_images
					
					if not dataset_root:
						Logger.LogError(f"Dataset root path not found in the yaml file ({path})")
						passed = False
					
					for dataset in all_images:
						datasetPath = os.path.join(ultralyticsDatasetsDir, dataset_root, dataset)
						if not os.path.isdir(datasetPath):
							Logger.LogError(f"Dataset path not found: {datasetPath}")
							passed = False
						elif VerifyFileCounts(datasetPath): passed = False

				except yaml.YAMLError as exc:
					Logger.LogError(f"Error reading yaml file ({path}): {exc}")
					passed = False

	

	# Verify dataset path


	Logger.LogInfo(f"Dataset directory verified: {ultralyticsDatasetsDir}")
	return passed
