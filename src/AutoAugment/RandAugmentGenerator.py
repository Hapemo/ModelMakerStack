import configparser
import random
from Logger import Logger
from Augmenter import Augmenter

globalN = None
globalAugmentationList = None

def ApplyRandAugmentMagnitude(augmentationConfig, augmentFinderConfig, M):
    '''
    Apply RandAugment to the dataset with the given policy, intensity, and level.
    Need to monkey patch the dataset class to apply the augmentations.
    '''
    global globalAugmentationList
    # Select augmentation operations to use
    # Select the magnitude of the operations
    Logger.LogCustom("RandAugment", f"Applying RandAugment with M = {M}")
    currentConfig = configparser.ConfigParser(inline_comment_prefixes="#")
    currentConfig.optionxform = str
    currentConfig.read_dict(augmentationConfig)

    # initializing the configs for albumentation and get a list of augmentations 
    augmenter = Augmenter()
    augmenter.CustomAugmentConfig(currentConfig, augmentFinderConfig, M)
    globalAugmentationList = augmenter.PrepAugments(currentConfig)


    Logger.LogCustom("RandAugment", f"Augmentations Initialized: {globalAugmentationList}")
    
def GetRandomAugments(n = None, augmentationList = None):
    ''' This function randomly selects N augmentations from the globalAugmentationList and returns them. '''
    # print("globalAugmentationList: ", globalAugmentationList)
    # print("augmentationList: ", augmentationList)

    if augmentationList is None or n is None:
        global globalN
        global globalAugmentationList
        if len(globalAugmentationList) < n:
            Logger.LogError(f"There are not enough augmentations to select from, please increase the number of augmentations, or decrease value of n. globalAugmentationList: {globalAugmentationList}, n: {n}")
            raise ValueError("Number of augmentations < n")
        selected_augmentations = random.sample(globalAugmentationList, globalN)
    else:
        if len(augmentationList) < n:
            Logger.LogError(f"There are not enough augmentations to select from, please increase the number of augmentations, or decrease value of n. augmentationList: {augmentationList}, n: {n}")
            raise ValueError("Number of augmentations < n")
        selected_augmentations = random.sample(augmentationList, n)

    return selected_augmentations




