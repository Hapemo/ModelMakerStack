''' This script is responsible for running the auto trainer with the given configuration files. '''

def main(mainConfig):
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(__file__))) # file's directory
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../AutoTrain'))) # AutoAugment's directory
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # go back one directory from file's directory
    import configparser
    from RunAutoTrain import RunAutoTrain, VerifyDatasetPath
    from Logger import Logger

    try:
        experimentFilePath = mainConfig["AutoTrainer"]["experimentFilePath"]
        experimentConfig = configparser.ConfigParser(inline_comment_prefixes="#")
        experimentConfig.optionxform = str
        experimentConfig.read(experimentFilePath)
        configFiles = [item.strip() for item in experimentConfig["General"]["configFiles"].split(",")]
    except Exception as e:
        print(f"Error reading experiment file ({experimentFilePath}): ", e)
        exit(1)

    result = VerifyDatasetPath(mainConfig["AutoTrainer"]["ultralyticsSettingPath"], 
                                                        experimentConfig["General"]["configFileDirectory"], 
                                                            configFiles, experimentFilePath)

    Logger.LogInfo("VerifyDatasetPath result: " + ("pass" if result else "fail"))
    if not result:
        Logger.LogError("VerifyDatasetPath failed, exiting...")
        exit(0)

    # Run the AutoTrain
    RunAutoTrain(experimentConfig["General"]["configFileDirectory"], configFiles, experimentConfig["General"]["name"])

# if __name__ == "__main__":
#     main()
