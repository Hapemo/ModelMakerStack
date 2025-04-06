''' This script takes care of deciding which operation to do. '''
import configparser
# Change this to your desired operation.
# - rand_augment_finder
# - auto_trainer
# - do_augment
# - test_extreme_augment
# - view_images
# - augment_tweaker
# operation = "rand_augment_finder"


if __name__ == "__main__":

    try:
        mainConfig = configparser.ConfigParser(inline_comment_prefixes="#")
        mainConfig.optionxform = str
        mainConfig.read("main.conf")
    except Exception as e:
        print("Error reading main.conf: ", e)
        exit(1)
    
    operation = mainConfig["General"]["operation"]
    # Decide which operation to do.
    if operation == "auto_trainer": mod = "src.Ops.AutoTrainer"
    elif operation == "rand_augment_finder": mod = "src.Ops.AugmentFinder"
    elif operation == "do_augment": mod = "src.Ops.DoAugment"
    elif operation == "test_extreme_augment": mod = "src.Ops.TestExtremeAugment"
    elif operation == "view_images": mod = "src.Ops.ViewImages"
    elif operation == "augment_tweaker": mod = "src.Ops.AugmentTweaker"
    # Add more operations here.
    else: 	
        print("Invalid operation")
        exit(1)
    
    # Run the operation
    import importlib
    mod = importlib.import_module(mod)
    mod.main(mainConfig)












