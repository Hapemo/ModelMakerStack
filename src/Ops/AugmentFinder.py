''' This script is responsible for running a list of rand augment finder experiments. '''



def main(mainConfig):
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # go back one directory from file's directory
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../AutoAugment'))) # AutoAugment's directory
    from Logger import Logger
    from RandAugmentFinder import RandAugmentFinder
    
    Logger.Initialize()

    paths = [path.strip() for path in mainConfig["AugmentFinder"]["configPaths"].split(",")]
    listOfConfigs = paths

    for config in listOfConfigs:
        randAugmentFinder = RandAugmentFinder(config)
        randAugmentFinder.GridSearch()

# if __name__ == "__main__":
#     main()

