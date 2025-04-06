''' This script will contain the main class to parse augmentation config and apply augmentation to the dataset. '''

import configparser
from ultralytics.data.augment import Mosaic, RandomPerspective, CopyPaste, MixUp, Albumentations, Compose, LetterBox, Format
from ultralytics.utils.checks import check_version
from ultralytics.utils import colorstr
import albumentations as A
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cv2
import numpy as np
import traceback
import uuid
import time
from Logger import Logger
import sys

Logger.printConsole = True

''' If you want to add on more augmentations for auto augmentation, create a function similar to CustomizeAdvancedBlur(),
include the things you want to change in there. Then add the function pointer on in CustomAugmentConfig(). 
'''

class Augmenter():
	''' This class will parse augmentation config and apply augmentation to the dataset. 
	How to use this class?
	1. Create an instance of this class.
	2. Call PrepAugments(configPath) to parse the augmentation config.
	2.5. Call TestAugment(imgPath, saveDir) to test a single augmentation on a single image. (Optional)
	3. Call AugmentAndSave(imageDirectories, outputDirectory) to apply augmentation to the dataset.
	
	'''
	_transforms_ = []
	
	def __init__(self):
		''' This method will initialize the augmenter object. '''
		self.augmentList = []
	
	def ParseConfig(self, configPath):
		# Parse
		Logger.LogCustom("augmenter", f"Parsing augmentation config, configPath: {configPath}") 
		config = configparser.ConfigParser(inline_comment_prefixes="#")
		config.optionxform = str
		config.read(configPath)
		return config

	def PrepAugments(self, config):
		''' This method will parse the augmentation config. '''
		
		# Get the augmentation config
		augmentNames = list(dict(config["General"]).keys())
		
		Logger.LogCustom("augmenter", f"Augment names found when prepping augments: {augmentNames}") 

		augmentList = []
		for name in augmentNames:
			use = bool(int(config["General"][name]))
			if not use: continue

			Logger.LogCustom("augmenter", f"Prepping augment({name})")
			try:
				augment = self.AugmentSelector(name, config)
				if augment is not None: 
					augmentList.append(augment)
				
			except Exception as e:
				Logger.LogError(f"Error prepping augment({name}): {e}")
		
		Logger.LogCustom("augmenter", f"Prepped augmentations.")
		return augmentList
	
	def PrepCompose(self, configPath):
		''' This method will call PrepAugment and prepare the augmentations into a compose object. '''
		self.config = self.ParseConfig(configPath)
		self.augmentList = self.PrepAugments(self.config)
		self.compose = A.Compose(self.augmentList, bbox_params=A.BboxParams(clip = True, format='yolo', label_fields=['category_ids']))

	def TestAugment(self, imgPath, savePath):
		''' This method perform single augmentation on a single image for each augmentations. 
		Parameters:
			imgPath: str, the path to the image
			saveDir: str, the directory to save the augmented images
		'''
		# Read the image
		image = cv2.imread(imgPath)
		if image is None:
			Logger.LogError(f"Error reading image from {imgPath}")
			return

		# Apply each augmentation and save the result
		for idx, augment in enumerate(self.augmentList):
			try:
				compose = A.Compose([augment], bbox_params=A.BboxParams(clip = True, format='yolo', label_fields=['category_ids'])) # self.PrepCompose([augment])
				saveDir = os.path.dirname(savePath)
				basename = os.path.basename(savePath)
				splitname = basename.rsplit('.', 1)[0]
				newname = f"{splitname}_{type(augment).__name__}"
				
				Augmenter.AugmentImage(imgPath, compose, saveDir, newname)

			except Exception as e:
				Logger.LogError(f"Error applying augmentation {idx} to image {imgPath}: {e}")
		

	def PopulateImagePaths(self, imageDirectories: list[str]):
		''' This method will apply augmentation to the dataset, and save the augmentation to an external storage. '''
		Logger.LogCustom("augmenter", f"Populating image paths from {imageDirectories}.")
		imagePaths = []

		for directory in imageDirectories:
			Logger.LogCustom("augmenter", f"Searching for images in {directory}.")
			for root, _, files in os.walk(directory):
				for file in files:
					if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
						image_path = os.path.join(root, file)
						text_path = os.path.splitext(image_path)[0] + '.txt'
						if os.path.exists(text_path):
							imagePaths.append(image_path)

		Logger.LogCustom("augmenter", f"Found {len(imagePaths)} images with corresponding text files.")
		
		return imagePaths

	def AugmentAndSave(self, imageDirectories: list[str], outputDirectory: str, logInterval: int = 5):
		''' This method will apply augmentation to the dataset, and save the augmentation to an external storage. '''
		imagePaths = self.PopulateImagePaths(imageDirectories)
		
		start_time = time.time()
		total_images = len(imagePaths)
		Logger.LogCustom("augmenter", f"Starting augmentation of {total_images} images.")

		for idx, imagePath in enumerate(imagePaths):
			try:
				unique_name = os.path.splitext(os.path.basename(imagePath))[0] + f"_aug_{uuid.uuid4().hex}"
				Augmenter.AugmentImage(imagePath, self.compose, outputDirectory, unique_name)
				
				if (idx + 1) % logInterval == 0 or (idx + 1) == total_images:
					elapsed_time = time.time() - start_time
					images_left = total_images - (idx + 1)
					avg_time_per_image = elapsed_time / (idx + 1)
					estimated_time_left = avg_time_per_image * images_left
					
					Logger.LogCustom("augmenter", f"Processed {idx + 1}/{total_images} images. Estimated time left: {estimated_time_left:.2f} seconds.")
				
			except Exception as e:
				Logger.LogError(f"Error augmenting {imagePath}: {e}")
	
	# --- Helper ---
	def GetPair(self, str, type, retList = False):
		''' This method takes in a string of numbers separated by ',' and split it to a tuple or list. '''
		_type = list if retList else tuple

		if type == "int": 		return _type(map(int, str.split(",")))
		elif type == "float":	return _type(map(float, str.split(",")))
		
	def ParamMultiplier(self, param:str, mag:float, rightMostOnly=False, oddOnly=False, limit:tuple[float, float]=None):
		""" Apply multipler on parameters for the augmentation magnitude customization 
		The multipler is more like a percentage addition. 
		param = param + param*mag

		List of supported parameter types:
		Pair - multiply both values in the pair (used if want to increase range in negative and positive direction)
		Pair(Max) - multiply the max value in the pair (used if want to increase range in positive direction only)

		rightMostOnly: bool, if True, adjust right most value only.
		oddOnly: bool, if True, it will add 1 to the value to make it odd if it's even.
		limit: tuple[float, float], the limit to apply to the parameter after multiplication, (min, max)
		"""

		# tuple - Pair, Pair(Max)
		if ',' in param:
			# float or int in tuple
			type = "float" if '.' in param else "int"
			_list = self.GetPair(param, type, retList=True)
			# rightMostOnly
			if rightMostOnly: _list = _list[:-1] + [_list[-1] + mag]
			else: _list = [i + mag for i in _list]

			# adjust type int after multiplication
			if type == "int": _list = [int(i) for i in _list]

			# odd only
			if type == "int" and oddOnly:
				_list = [i+1 if i % 2 == 0 else i for i in _list]
				for i, val in enumerate(_list):
					if val % 2 == 0: _list[i] += 1
			
			# limit check
			if limit is not None:
				for val in _list: 
					if val < limit[0]: val = limit[0]
					if val > limit[1]: val = limit[1]

			return str(tuple(_list))[1:-1]

		# single value
		else:
			# float or int
			_type = float if '.' in param else int
			val = _type(_type(param) + mag)

			# odd only
			if _type == "int" and oddOnly:
				if val % 2 == 0: val += 1

			# limit check
			if limit is not None:
				for val in _list: 
					if val < limit[0]: val = limit[0]
					elif val > limit[1]: val = limit[1]
			
			if _type == float:
				Augmenter.RemoveLeadingZerosAndNine(val)

			return str(_type(val))
	
	def ParamLimiter(self, param, lowerLimit, upperLimit, incremental: bool = True):
		''' The type of the param and limit must be the same 
		incremental is used for tuple or list, to ensure the first value is smaller than the subsequent values
		'''
		# if type(param) != type(lowerLimit):
		# 	Logger.LogError(f"Error: ParamLimiter() param({param}) and limit({lowerLimit}) must be the same type.")
		# 	return None
		
		if (type(param) == int) or (type(param) == float):
			if param < lowerLimit: newParam = lowerLimit
			elif param > upperLimit: newParam = upperLimit
			else: newParam = param

			if type(newParam) == float: newParam = Augmenter.RemoveLeadingZerosAndNine(newParam)
			return type(param)(newParam)

		elif (type(param) == tuple) or (type(param) == list):
			# if len(param) != len(lowerLimit) or len(param) != len(upperLimit):
			# 	Logger.LogError(f"Error: ParamLimiter() param({param}) and limit({lowerLimit}, {upperLimit}) must have the same length.")
			# 	return None

			newParam = []
			if incremental:
				ogType = type(param)
				param = ogType(sorted(list(param)))

			if (type(lowerLimit) == int) or (type(lowerLimit) == float): lowerLimit = [lowerLimit] * len(param)
			if (type(upperLimit) == int) or (type(upperLimit) == float): upperLimit = [upperLimit] * len(param)
			for val, low, up in zip(param, lowerLimit, upperLimit):
				newParam.append(self.ParamLimiter(val, low, up))

			return type(param)(newParam)
		else:
			Logger.LogError(f"Error: ParamLimiter() param({param}) of type({type(param)}) is not supported.")
			return None

	def ReadBBFile(filepath:str, imgDimension: list[float]):
		''' Read bb in info file and convert it into albumentation's bb format 
		returns list of cls and list of bb, eg. [0,1], [[0,1,2,3],[4,5,6,7]]
		'''
		file = open(filepath, "r")
		if file.closed:
			print(f"{__name__} Error: {filepath} not found")
			return None, None

		catIDs = []
		bbList = []
		for line in file.readlines():
			splitInfo = line.split(" ")
			clsNum = int(splitInfo.pop(0))
			
			# bb = BBYoloToAlbumentations(splitInfo, width, height)
			bb = [float(i) for i in splitInfo]
			for i, coord in enumerate(bb):
				if coord > 1:
					print(f"{__name__} Error: {filepath} has bb coords greater than 1, setting it to 1...")
					bb[i] = 1
				if coord < 0:
					print(f"{__name__} Error: {filepath} has bb coords lesser than 0, setting it to 0...")
					bb[i] = 0

			# Reduce the bb by 1 pixel on each side to convert it to albumentations format
			bb[0] = bb[0]+10/imgDimension[0]
			bb[1] = bb[1]+10/imgDimension[1]
			bb[2] = bb[2]-10/imgDimension[0]
			bb[3] = bb[3]-10/imgDimension[1]

			catIDs.append(clsNum)
			bbList.append(bb)
		return catIDs,bbList

	def ReadImage(imgPath:str):
		''' The filename w/o extension of the image and bounding box text should be the same. '''
		bbTextPath = imgPath.replace(imgPath.split('.')[-1], "txt")

		try:
			img:np.ndarray = cv2.imread(imgPath)
		except Exception as e:
			print(f"{__name__} Error: Unable to cv2.imread {imgPath}")
			print(traceback.format_exc())
			return None, None, None
		height = img.shape[0]
		width = img.shape[1]
		catIDs, BBs = Augmenter.ReadBBFile(bbTextPath, [width, height])
		return img, catIDs, BBs

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

	# --- Augmentations ---
	@staticmethod
	def AugmentImage(imgPath:str, compose:A.Compose, imgSaveDirectory:str=None, imgSaveName:str=None):
		if not os.path.isdir(imgSaveDirectory): os.makedirs(imgSaveDirectory)
		
		image, category_ids, bboxes = Augmenter.ReadImage(imgPath)
		if image is None or category_ids is None or bboxes is None:
			print(f"{__name__} Error: unable to ReadImage {imgPath}")
			return
		try:
			textPath = os.path.join(imgSaveDirectory, imgSaveName + ".txt")
			textFile = open(textPath, "w")
		except Exception as e:
			print(f"{__name__} Error: unable to open {textPath} for textFile\n{e}")
			return None
		
		try:
			transformed = compose(image=image, bboxes=bboxes, category_ids=category_ids)
		except Exception as e:
			print(f"{__name__} Error: unable to compose {imgPath}, compose: {compose}, bboxes: {bboxes}, category_ids: {category_ids}")
			print("error: ", traceback.format_exc())
			return

		msg = ""
		for bb, cat in zip(transformed['bboxes'], transformed['category_ids']):
			msg += f"{int(cat)} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n"
		
		if imgSaveDirectory is None or imgSaveName is None:
			return transformed['image']
		
		textFile.write(msg)
		imgSavePath = os.path.join(imgSaveDirectory, imgSaveName + ".jpg")
		cv2.imwrite(imgSavePath, transformed['image'])
		# print(f"Saved augmented image to {imgSavePath}")
	
	def CustomAugmentConfig(self, config:configparser.ConfigParser, magConfig: configparser.ConfigParser, mag:float):
		"""
		Creates a new configuration for albumentations based on the magnitude and probability multipliers.
		Each augmentation operation has different parameter rules, some require int and some require odd numbers.
		So you will have to create a custom function for each augmentation parameter customization.

		This is used for initializing each experiment run of randaugment.
		Args:
			config (configparser.ConfigParser): The configuration parser object to be modified.
			num (int): The number of augmentations to apply.
			mag (float): The magnitude multiplier for the augmentations.
		Returns:
			None
		"""
		# Apply the custom augmentation config
		augmentNames = list(dict(config["General"]).keys())

		Logger.LogCustom("augmenter", f"Augment names found when customizing augments: {augmentNames}") 

		for name in augmentNames:

			# prep configs
			if name in magConfig:
				subMagConfig = magConfig[name]
				subMagConfig = { key: float(subMagConfig[key]) for key in subMagConfig }
			else: subMagConfig = None # this can happen when there is no magConfig for the augmentation, because it doesn't have any parameter to tweek.

			fptr = None
			if 	 name == "AdvancedBlur":			fptr = self.CustomizeAdvancedBlur
			elif name == "Brightness":			fptr = self.CustomizeBrightness
			elif name == "Contrast":			fptr = self.CustomizeContrast
			elif name == "Saturation":			fptr = self.CustomizeSaturation
			elif name == "Hue":					fptr = self.CustomizeHue
			elif name == "Emboss": 				fptr = self.CustomizeEmboss
			elif name == "GaussNoise": 			fptr = self.CustomizeGaussNoise
			elif name == "ISONoise": 			fptr = self.CustomizeISONoise
			elif name == "MultiplicativeNoise": fptr = self.CustomizeMultiplicativeNoise
			elif name == "RandomFog": 			fptr = self.CustomizeRandomFog
			elif name == "RandomShadow": 		fptr = self.CustomizeRandomShadow
			elif name == "RandomToneCurve": 	fptr = self.CustomizeRandomToneCurve
			elif name == "RingingOvershoot": 	fptr = self.CustomizeRingingOvershoot
			elif name == "PixelDropout": 		fptr = self.CustomizePixelDropout
			elif name == "AffineTranslate": 	fptr = self.CustomizeAffineTranslate
			elif name == "AffineScale": 		fptr = self.CustomizeAffineScale
			elif name == "AffineShear": 		fptr = self.CustomizeAffineShear
			elif name == "RandomGridShuffle": 	fptr = self.CustomizeRandomGridShuffle
			elif name == "CoarseDropout": 		fptr = self.CustomizeCoarseDropout
			elif name == "Perspective": 		fptr = self.CustomizePerspective
			elif name == "SafeRotate": 			fptr = self.CustomizeSafeRotate
			else: 
				Logger.LogError(f"Error: CustomAugmentConfig() does not support augment({name}). Maybe it's not implemented yet in code, or the augmentation name is wrong in config file.")
				continue
			Logger.LogInfo(f"CustomAugmentConfig(): Applying custom augment({name})")
			fptr(config[name], subMagConfig, mag)
			config[name]["p"] = "1.0"

		Logger.LogInfo(f"CustomAugmentConfig(): Customized the augmentation config.")

	def AugmentSelector(self, name:str, config: configparser.ConfigParser):
		"""
		Selects the augmentation based on name and config.
		Parameters:
			name (str): The name of the augmentation to be applied. Supported values are "CLAHE", "AdvancedBlur", and "ColorJitter".
			config (configparser.ConfigParser): Configuration parameters required for the selected augmentation.
		Returns:
			None
		Raises:
			Exception: If there is an error in preparing the augmentation, it logs the error message.
		"""
		
		try:
			if 	 name == "CLAHE": 				return self.PrepCLAHE(config)
			elif name == "AdvancedBlur": 		return self.PrepAdvancedBlur(config)
			elif name == "Brightness": 			return self.PrepBrightness(config)
			elif name == "Contrast": 			return self.PrepContrast(config)
			elif name == "Saturation": 			return self.PrepSaturation(config)
			elif name == "Hue": 				return self.PrepHue(config)
			elif name == "D4": 					return self.PrepD4(config)
			elif name == "Emboss": 				return self.PrepEmboss(config)
			elif name == "GaussNoise": 			return self.PrepGaussNoise(config)
			elif name == "ISONoise": 			return self.PrepISONoise(config)
			elif name == "MultiplicativeNoise": return self.PrepMultiplicativeNoise(config)
			elif name == "RandomFog": 			return self.PrepRandomFog(config)
			elif name == "RandomShadow": 		return self.PrepRandomShadow(config)
			elif name == "RandomToneCurve": 	return self.PrepRandomToneCurve(config)
			elif name == "RingingOvershoot": 	return self.PrepRingingOvershoot(config)
			elif name == "PixelDropout": 		return self.PrepPixelDropout(config)
			elif name == "AffineTranslate": 	return self.PrepAffineTranslate(config)
			elif name == "AffineScale": 		return self.PrepAffineScale(config)
			elif name == "AffineShear": 		return self.PrepAffineShear(config)
			elif name == "RandomGridShuffle": 	return self.PrepRandomGridShuffle(config)
			elif name == "CoarseDropout": 		return self.PrepCoarseDropout(config)
			elif name == "Perspective": 		return self.PrepPerspective(config)
			elif name == "SafeRotate": 			return self.PrepSafeRotate(config)
			else:
				print(f"Augment({name}) is not supported")
		  
		except Exception as e:
			Logger.LogError(f"Error selecting augment({name}): {traceback.format_exc()}")
			return None

	# --- CLAHE ---
	def PrepCLAHE(self, config: configparser.ConfigParser):
		''' This method will parse the CLAHE config. '''
		p = float(config["CLAHE"].get("p", 0.5))
		return A.CLAHE(p=p)

	def PrepAdvancedBlur(self, config: configparser.ConfigParser):
		''' This method will parse the AdvancedBlur config. '''
		blur_limit = 	self.GetPair(config["AdvancedBlur"].get("blur_limit", "3,7"), "int")
		sigma_x_limit = self.GetPair(config["AdvancedBlur"].get("sigma_x_limit", "0.2,1.0"), "float")
		sigma_y_limit = self.GetPair(config["AdvancedBlur"].get("sigma_y_limit", "0.2,1.0"), "float")
		rotate_limit = 	self.GetPair(config["AdvancedBlur"].get("rotate_limit", "-90,90"), "int")
		# beta_limit = 	self.GetPair(config["AdvancedBlur"].get("beta_limit", "0.5,8.0"), "float")
		noise_limit = 	self.GetPair(config["AdvancedBlur"].get("noise_limit", "0.9,1.1"), "float")
		p = float(config["AdvancedBlur"].get("p", 0.5)) 

		self.ParamLimiter(blur_limit, -float('inf'), float('inf'), incremental=False)
		self.ParamLimiter(sigma_x_limit, 0, float('inf'), incremental=False)
		self.ParamLimiter(sigma_y_limit, 0, float('inf'), incremental=False)
		self.ParamLimiter(rotate_limit, -180, 180, incremental=False)
		# self.ParamLimiter(beta_limit, 0, float('inf'), incremental=False)
		self.ParamLimiter(noise_limit, 0, float('inf'), incremental=False)


		return A.AdvancedBlur(blur_limit=blur_limit, sigma_x_limit=sigma_x_limit, sigma_y_limit=sigma_y_limit, rotate_limit=rotate_limit, noise_limit=noise_limit, p=p)
	def CustomizeAdvancedBlur(self, config: configparser.SectionProxy, magConfig: dict, mag:float):
		''' This method will customize the AdvancedBlur config. '''
		if bool(magConfig["blur_limit"]): 		config["blur_limit"] 	= self.ParamMultiplier(config["blur_limit"], 	magConfig["blur_limit"] * mag, oddOnly=True)
		if bool(magConfig["sigma_x_limit"]):	config["sigma_x_limit"] = self.ParamMultiplier(config["sigma_x_limit"], magConfig["sigma_x_limit"] * mag)
		if bool(magConfig["sigma_y_limit"]):	config["sigma_y_limit"] = self.ParamMultiplier(config["sigma_y_limit"], magConfig["sigma_y_limit"] * mag)
		if bool(magConfig["rotate_limit"]): 	config["rotate_limit"] 	= self.ParamMultiplier(config["rotate_limit"], 	magConfig["rotate_limit"] * mag, rightMostOnly=True)
		if bool(magConfig["noise_limit"]): 		config["noise_limit"] 	= self.ParamMultiplier(config["noise_limit"], 	magConfig["noise_limit"] * mag)

	def PrepBrightness(self, config: configparser.ConfigParser):
		brightness = self.GetPair(config["Brightness"].get("brightness", "1,1"), "float")
		p = float(config["Brightness"].get("p", 0.5))
		brightness = self.ParamLimiter(brightness, 0, float('inf'))
		return A.ColorJitter(brightness=brightness, contrast=(1,1), saturation=(1,1), hue=(0,0), p=p)
	def CustomizeBrightness(self, config: configparser.ConfigParser, magConfig: dict, mag:float):
		if bool(magConfig["brightness"]): 		config["brightness"] 	= self.ParamMultiplier(config["brightness"], 	magConfig["brightness"] * mag)
		pass

	def PrepContrast(self, config: configparser.ConfigParser):
		contrast = self.GetPair(config["Contrast"].get("contrast", "1,1"), "float")
		p = float(config["Contrast"].get("p", 0.5))
		contrast = self.ParamLimiter(contrast, 0, float('inf'))
		return A.ColorJitter(brightness=(1,1), contrast=contrast, saturation=(1,1), hue=(0,0), p=p)
	def CustomizeContrast(self, config: configparser.ConfigParser, magConfig: dict, mag:float):
		if bool(magConfig["contrast"]): 		config["contrast"] 	= self.ParamMultiplier(config["contrast"], 	magConfig["contrast"] * mag)
		pass
	
	def PrepSaturation(self, config: configparser.ConfigParser):
		saturation = self.GetPair(config["Saturation"].get("saturation", "1,1"), "float")
		p = float(config["Saturation"].get("p", 0.5))
		saturation = self.ParamLimiter(saturation, 0, float('inf'))
		return A.ColorJitter(brightness=(1,1), contrast=(1,1), saturation=saturation, hue=(0,0), p=p)
	def CustomizeSaturation(self, config: configparser.ConfigParser, magConfig: dict, mag:float):
		if bool(magConfig["saturation"]): 		config["saturation"] 	= self.ParamMultiplier(config["saturation"], 	magConfig["saturation"] * mag)
		pass
	
	def PrepHue(self, config: configparser.ConfigParser):
		hue = self.GetPair(config["Hue"].get("hue", "0,0"), "float")
		p = float(config["Hue"].get("p", 0.5))
		hue = self.ParamLimiter(hue, -0.5, 0.5)
		return A.ColorJitter(brightness=(1,1), contrast=(1,1), saturation=(1,1), hue=hue, p=p)
	def CustomizeHue(self, config: configparser.ConfigParser, magConfig: dict, mag:float):
		if bool(magConfig["hue"]): 		config["hue"] 	= self.ParamMultiplier(config["hue"], 	magConfig["hue"] * mag)

	def PrepD4(self, config: configparser.ConfigParser):
		''' This method will parse the CLAHE config. '''
		p = float(config["D4"].get("p", 0.5))

		return A.D4(p=p)

	def PrepEmboss(self, config: configparser.ConfigParser):
		''' This method will parse the Emboss config. '''
		alpha = 	self.GetPair(config["Emboss"].get("alpha", "0.2, 0.5"), "float")
		strength = 	self.GetPair(config["Emboss"].get("strength", "0.2, 0.7"), "float")
		p = float(config["Emboss"].get("p", 0.5)) 

		alpha = self.ParamLimiter(alpha, 0, 1)
		strength = self.ParamLimiter(strength, 0, float('inf'))

		return A.Emboss(alpha=alpha, strength=strength, p=p)
	def CustomizeEmboss(self, config: configparser.SectionProxy, magConfig: dict, mag:float):
		''' This method will customize the Emboss config. '''
		if bool(magConfig["alpha"]): 		config["alpha"] 	= self.ParamMultiplier(config["alpha"], magConfig["alpha"] * mag)
		if bool(magConfig["strength"]):		config["strength"]	= self.ParamMultiplier(config["strength"], magConfig["strength"] * mag)

	def PrepGaussNoise(self, config: configparser.ConfigParser):
		''' This method will parse the GaussNoise config. '''
		var_limit = self.GetPair(config["GaussNoise"].get("var_limit", "10.0, 50.0"), "float")
		p = float(config["GaussNoise"].get("p", 0.5)) 

		var_limit = self.ParamLimiter(var_limit, 0, float('inf'))

		return A.GaussNoise(var_limit=var_limit, p=p)
	def CustomizeGaussNoise(self, config: configparser.SectionProxy, magConfig: dict, mag:float):
		''' This method will customize the GaussNoise config. '''
		if bool(magConfig["var_limit"]): 	config["var_limit"] = self.ParamMultiplier(config["var_limit"], magConfig["var_limit"] * mag)

	def PrepISONoise(self, config: configparser.ConfigParser):
		''' This method will parse the ISONoise config. '''
		color_shift = 	self.GetPair(config["ISONoise"].get("color_shift", "0.01, 0.05"), "float")
		intensity = 	self.GetPair(config["ISONoise"].get("intensity", "0.1, 0.5"), "float")
		p = float(config["ISONoise"].get("p", 0.5)) 

		color_shift = self.ParamLimiter(color_shift, 0, 1)
		intensity = self.ParamLimiter(intensity, 0, float('inf'))

		return A.ISONoise(color_shift=color_shift, intensity=intensity, p=p)
	def CustomizeISONoise(self, config: configparser.SectionProxy, magConfig: dict, mag:float):
		''' This method will customize the ISONoise config. '''
		if bool(magConfig["color_shift"]):  config["color_shift"] = self.ParamMultiplier(config["color_shift"], magConfig["color_shift"] * mag)
		if bool(magConfig["intensity"]):    config["intensity"] = self.ParamMultiplier(config["intensity"], magConfig["intensity"] * mag)

	def PrepMultiplicativeNoise(self, config: configparser.ConfigParser):
		''' This method will parse the MultiplicativeNoise config. '''
		multiplier = self.GetPair(config["MultiplicativeNoise"].get("multiplier", "0.9, 1.1"), "float")
		p = float(config["MultiplicativeNoise"].get("p", 0.5))

		multiplier = self.ParamLimiter(multiplier, 0, float('inf'))

		return A.MultiplicativeNoise(multiplier=multiplier, per_channel=True, elementwise=True, p=p)
	def CustomizeMultiplicativeNoise(self, config: configparser.SectionProxy, magConfig: dict, mag: float):
		''' This method will customize the MultiplicativeNoise config. '''
		if bool(magConfig["multiplier"]):	config["multiplier"] = self.ParamMultiplier(config["multiplier"], magConfig["multiplier"] * mag)

	def PrepRandomFog(self, config: configparser.ConfigParser):
		''' This method will parse the RandomFog config. '''
		fog_coef_range = self.GetPair(config["RandomFog"].get("fog_coef_range", "0.3, 1.0"), "float")
		alpha_coef = float(config["RandomFog"].get("alpha_coef", "0.08"))
		p = float(config["RandomFog"].get("p", 0.5))

		fog_coef_range = self.ParamLimiter(fog_coef_range, 0, 1)
		alpha_coef = self.ParamLimiter(alpha_coef, 0, 1)

		return A.RandomFog(fog_coef_range=fog_coef_range, alpha_coef=alpha_coef, p=p)
	def CustomizeRandomFog(self, config: configparser.SectionProxy, magConfig: dict, mag: float):
		''' This method will customize the RandomFog config. '''
		if bool(magConfig["fog_coef_range"]): config["fog_coef_range"] = self.ParamMultiplier(config["fog_coef_range"], magConfig["fog_coef_range"] * mag)
		if bool(magConfig["alpha_coef"]): config["alpha_coef"] = self.ParamMultiplier(config["alpha_coef"], magConfig["alpha_coef"] * mag)

	def PrepRandomShadow(self, config: configparser.ConfigParser):
		''' This method will parse the RandomShadow config. '''
		num_shadows_limit = self.GetPair(config["RandomShadow"].get("num_shadows_limit", "1, 2"), "int")
		shadow_dimension = int(config["RandomShadow"].get("shadow_dimension", 5))
		p = float(config["RandomShadow"].get("p", 0.5))

		num_shadows_limit = self.ParamLimiter(num_shadows_limit, 1, float('inf'))
		shadow_dimension = self.ParamLimiter(shadow_dimension, 3, float('inf'))

		return A.RandomShadow(num_shadows_limit=num_shadows_limit, shadow_roi=(0,0,1,1), shadow_dimension=shadow_dimension, p=p)
	def CustomizeRandomShadow(self, config: configparser.SectionProxy, magConfig: dict, mag: float):
		''' This method will customize the RandomShadow config. '''
		if bool(magConfig["num_shadows_limit"]): config["num_shadows_limit"] = self.ParamMultiplier(config["num_shadows_limit"], magConfig["num_shadows_limit"] * mag)
		if bool(magConfig["shadow_dimension"]): config["shadow_dimension"] = self.ParamMultiplier(config["shadow_dimension"], magConfig["shadow_dimension"] * mag)

	def PrepRandomToneCurve(self, config: configparser.ConfigParser):
		''' This method will parse the RandomToneCurve config. '''
		scale = float(config["RandomToneCurve"].get("scale", 0.1))
		p = float(config["RandomToneCurve"].get("p", 0.5))

		scale = self.ParamLimiter(scale, 0, 1)

		return A.RandomToneCurve(scale=scale, p=p)
	def CustomizeRandomToneCurve(self, config: configparser.SectionProxy, magConfig: dict, mag: float):
		''' This method will customize the RandomToneCurve config. '''
		if bool(magConfig["scale"]): config["scale"] = self.ParamMultiplier(config["scale"], magConfig["scale"] * mag)

	def PrepRingingOvershoot(self, config: configparser.ConfigParser):
		''' This method will parse the RingingOvershoot config. '''
		blur_limit = self.GetPair(config["RingingOvershoot"].get("blur_limit", "7, 15"), "int")
		cutoff = self.GetPair(config["RingingOvershoot"].get("cutoff", "0.7854, 1.57"), "float")
		p = float(config["RingingOvershoot"].get("p", 0.5))

		blur_limit = self.ParamLimiter(blur_limit, 3, float('inf'))
		cutoff = self.ParamLimiter(cutoff, 0, np.pi)

		return A.RingingOvershoot(blur_limit=blur_limit, cutoff=cutoff, p=p)
	def CustomizeRingingOvershoot(self, config: configparser.SectionProxy, magConfig: dict, mag: float):
		''' This method will customize the RingingOvershoot config. '''
		if bool(magConfig["blur_limit"]): config["blur_limit"] = self.ParamMultiplier(config["blur_limit"], magConfig["blur_limit"] * mag, oddOnly=True)
		if bool(magConfig["cutoff"]): config["cutoff"] = self.ParamMultiplier(config["cutoff"], magConfig["cutoff"] * mag)

	def PrepPixelDropout(self, config: configparser.ConfigParser):
		''' This method will parse the PixelDropout config. '''
		dropout_prob = float(config["PixelDropout"].get("dropout_prob", 0.01))
		p = float(config["PixelDropout"].get("p", 0.5))

		dropout_prob = self.ParamLimiter(dropout_prob, 0, 1)

		return A.PixelDropout(dropout_prob=dropout_prob, p=p)
	def CustomizePixelDropout(self, config: configparser.SectionProxy, magConfig: dict, mag: float):
		''' This method will customize the PixelDropout config. '''
		if bool(magConfig["dropout_prob"]): config["dropout_prob"] = self.ParamMultiplier(config["dropout_prob"], magConfig["dropout_prob"] * mag)

	def PrepAffineTranslate(self, config: configparser.ConfigParser):
		''' This method will parse the Affine (Translate) config. '''
		translate_percent = self.GetPair(config["AffineTranslate"].get("translate_percent", "-0.1,0.1"), "float")
		p = float(config["AffineTranslate"].get("p", 0.5))

		translate_percent = self.ParamLimiter(translate_percent, -1, 1)

		return A.Affine(translate_percent=translate_percent, p=p)
	def CustomizeAffineTranslate(self, config: configparser.SectionProxy, magConfig: dict, mag: float):
		''' This method will customize the Affine (Translate) config. '''
		if bool(magConfig["translate_percent"]): config["translate_percent"] = self.ParamMultiplier(config["translate_percent"], magConfig["translate_percent"] * mag)

	def PrepAffineScale(self, config: configparser.ConfigParser):
		''' This method will parse the Affine (Scale) config. '''
		scale = self.GetPair(config["AffineScale"].get("scale", "0.8,1.2"), "float")
		p = float(config["AffineScale"].get("p", 0.5))

		return A.Affine(scale=scale, keep_ratio=True, p=p)
	def CustomizeAffineScale(self, config: configparser.SectionProxy, magConfig: dict, mag: float):
		''' This method will customize the Affine (Scale) config. '''
		if bool(magConfig["scale"]): config["scale"] = self.ParamMultiplier(config["scale"], magConfig["scale"] * mag)

	def PrepAffineShear(self, config: configparser.ConfigParser):
		''' This method will parse the Affine (Shear) config. '''
		shear = self.GetPair(config["AffineShear"].get("shear", "-0.1,0.1"), "float")
		p = float(config["AffineShear"].get("p", 0.5))

		return A.Affine(shear=shear, p=p)
	def CustomizeAffineShear(self, config: configparser.SectionProxy, magConfig: dict, mag: float):
		''' This method will customize the Affine (Shear) config. '''
		if bool(magConfig["shear"]): config["shear"] = self.ParamMultiplier(config["shear"], magConfig["shear"] * mag)

	def PrepRandomGridShuffle(self, config: configparser.ConfigParser):
		''' This method will parse the RandomGridShuffle config. '''
		grid = self.GetPair(config["RandomGridShuffle"].get("grid", "3,3"), "int")
		p = float(config["RandomGridShuffle"].get("p", 0.5))

		grid = self.ParamLimiter(grid, 2, float('inf'))

		return A.RandomGridShuffle(grid=grid, p=p)
	def CustomizeRandomGridShuffle(self, config: configparser.SectionProxy, magConfig: dict, mag: float):
		''' This method will customize the RandomGridShuffle config. '''
		if bool(magConfig["grid"]): config["grid"] = self.ParamMultiplier(config["grid"], magConfig["grid"] * mag)

	def PrepCoarseDropout(self, config: configparser.ConfigParser):
		''' This method will parse the CoarseDropout config. '''
		num_holes_range = self.GetPair(config["CoarseDropout"].get("num_holes_range", "3,5"), "int")
		hole_height_range = self.GetPair(config["CoarseDropout"].get("hole_height_range", "8,8"), "int")
		hole_width_range = self.GetPair(config["CoarseDropout"].get("hole_width_range", "8,8"), "int")
		p = float(config["CoarseDropout"].get("p", 0.5))

		num_holes_range = self.ParamLimiter(num_holes_range, 1, float('inf'))
		hole_height_range = self.ParamLimiter(hole_height_range, 1, float('inf'))
		hole_width_range = self.ParamLimiter(hole_width_range, 1, float('inf'))

		return A.CoarseDropout(num_holes_range=num_holes_range, hole_height_range=hole_height_range, hole_width_range=hole_width_range, p=p)
	def CustomizeCoarseDropout(self, config: configparser.SectionProxy, magConfig: dict, mag: float):
		''' This method will customize the CoarseDropout config. '''
		if bool(magConfig["num_holes_range"]): config["num_holes_range"] = self.ParamMultiplier(config["num_holes_range"], magConfig["num_holes_range"] * mag)
		if bool(magConfig["hole_height_range"]): config["hole_height_range"] = self.ParamMultiplier(config["hole_height_range"], magConfig["hole_height_range"] * mag)
		if bool(magConfig["hole_width_range"]): config["hole_width_range"] = self.ParamMultiplier(config["hole_width_range"], magConfig["hole_width_range"] * mag)

	def PrepPerspective(self, config: configparser.ConfigParser):
		''' This method will parse the Perspective config. '''
		scale = self.GetPair(config["Perspective"].get("scale", "0.05, 0.1"), "float")
		p = float(config["Perspective"].get("p", 0.5))

		scale = self.ParamLimiter(scale, 0, float('inf'))

		return A.Perspective(scale=scale, keep_size=True, fit_output=True, p=p)
	def CustomizePerspective(self, config: configparser.SectionProxy, magConfig: dict, mag: float):
		''' This method will customize the Perspective config. '''
		if bool(magConfig["scale"]): config["scale"] = self.ParamMultiplier(config["scale"], magConfig["scale"] * mag)

	def PrepSafeRotate(self, config: configparser.ConfigParser):
		''' This method will parse the SafeRotate config. '''
		limit = self.GetPair(config["SafeRotate"].get("limit", "-45,45"), "int")
		p = float(config["SafeRotate"].get("p", 0.5))

		limit = self.ParamLimiter(limit, -180, 180)

		return A.SafeRotate(limit=limit, p=p)
	def CustomizeSafeRotate(self, config: configparser.SectionProxy, magConfig: dict, mag: float):
		''' This method will customize the SafeRotate config. '''
		if bool(magConfig["limit"]): config["limit"] = self.ParamMultiplier(config["limit"], magConfig["limit"] * mag)

# augmenter = Augmenter()
# augmenter.PrepCompose("Augment.conf")
# augmenter.TestAugment("C:/Users/TeohJ/Desktop/Data Prep/datasets/OnePieceFlow/Original/OnePieceFlowImg (2).jpg", "test")
# augmenter.AugmentAndSave(["C:/Users/sinmatrix01/Desktop/FlexiVisionSystem/labels"], "C:/Users/sinmatrix01/Desktop/FlexiVisionSystem/bat7_augmented")

# augmenter.PrepConfig("Augment.conf")
# print("Before: ", augmenter.config["CLAHE"]["p"])
# augmenter.CustomizeCLAHE(augmenter.config, 0, 0.001)
# print("After: ", augmenter.config["CLAHE"]["p"])


# augmentConfig = augmenter.ParseConfig("Augment.conf")
# magConfig = augmenter.ParseConfig("AutoAugment.conf")
# augmenter.CustomAugmentConfig(augmentConfig, magConfig, 10, 0.1)

# for key in augmentConfig:
#   print(key)
#   print(dict(augmentConfig[key]))


# from torchvision.transforms import RandAugment

