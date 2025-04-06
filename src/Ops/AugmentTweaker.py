''' a system that lets you change the value of each augment parameter on the fly and see the values reflected. 
Create minimum and maximum augmentation parameters and generate a config file for augment finder and augment.

Note: the minimum augmentation parameter will be the starting point and the maximum augmentation parameter will be the end point.
The values in between will be calculated by the formula: a(i) = a + a*mag(i)*mod
The starting point is based on mag(i) = 0. So if your starting point is 1, then a(i) = a.
The ending point is based on mag(i) = M_max that you inserted. So if your ending point is 2, then a(i) = a + a*mod

Controls:
    [ and ]: Switch between augmentation types.
    up and down: Increase or decrease the current parameter value.
    left and right: Switch between parameters of the current augmentation.
    p: Print the current value of the selected parameter.
    - and =: Decrease or increase the adjustment step size.
    n: Save the current augmentation parameters as the minimum configuration.
    m: Save the current augmentation parameters as the maximum configuration.
    b: Generate and save the augmentation and finder configuration files.
    esc: Exit the program.

usage:
    1. Adjust the image augmentation parameters to your liking 
'''
import albumentations as A
import os
import cv2
from matplotlib import pyplot as plt
import keyboard
import copy

augment_parameters = {
    "AdvancedBlur": {
        "class": A.AdvancedBlur,
        "params": {
            "blur_limit": [3, 3],
            "sigma_x_limit": [2.0, 2.0],
            "sigma_y_limit": [2.0, 2.0],
            "rotate_limit": [45, 45],
            "noise_limit": [0.1, 0.1],
        }
    },
    "ColorJitter": {
        "class": A.ColorJitter,
        "params": {
            "brightness": [0.8, 0.8],
            "contrast": [0.8, 0.8],
            "saturation": [0.8, 0.8],
            "hue": [0.1, 0.1],
        }
    },
    "Emboss": {
        "class": A.Emboss,
        "params": {
            "alpha": [1.0, 1.0],
            "strength": [1.5, 1.5],
        }
    },
    "GaussNoise": {
        "class": A.GaussNoise,
        "params": {
            "var_limit": [0.1, 0.1],
        }
    },
    "ISONoise": {
        "class": A.ISONoise,
        "params": {
            "color_shift": [0.05, 0.05],
            "intensity": [0.5, 0.5],
        }
    },
    "MultiplicativeNoise": {
        "class": A.MultiplicativeNoise,
        "params": {
            "multiplier": [1.1, 1.1],
        }
    },
    "RandomFog": {
        "class": A.RandomFog,
        "params": {
            "fog_coef_range": [0.5, 0.5],
            "alpha_coef": 0.3,
        }
    },
    "RandomShadow": {
        "class": A.RandomShadow,
        "params": {
            "num_shadows_limit": [1, 1],
            "shadow_dimension": 5,
        }
    },
    "RandomToneCurve": {
        "class": A.RandomToneCurve,
        "params": {
            "scale": 1.0,
        }
    },
    "RingingOvershoot": {
        "class": A.RingingOvershoot,
        "params": {
            "blur_limit": [7, 7],
            "cutoff": [0.5, 0.5],
        }
    },
    "PixelDropout": {
        "class": A.PixelDropout,
        "params": {
            "dropout_prob": 0.1,
        }
    },
    "CLAHE": {  
        "class": A.CLAHE,
        "params": {}
    },
    "Affine": {
        "class": A.Affine,
        "params": {
            "translate_percent": [0.0, 0.0],
            "scale": [1.0, 1.0],
            "shear": [0.0, 0.0],
        }
    },
    "RandomGridShuffle": {
        "class": A.RandomGridShuffle,
        "params": {
            "grid": [2, 2]
        }
    },
    "CoarseDropout": {
        "class": A.CoarseDropout,
        "params": {
            "num_holes_range" : [8, 8],
            "hole_height_range" : [20, 20],
            "hole_width_range" : [20, 20]
        }
    },
    "Perspective": {
        "class": A.Perspective,
        "params": {
            "scale": [0.1, 0.1],
        }
    },
    "SafeRotate": {
        "class": A.SafeRotate,
        "params": {
            "limit": [60, 60],
        }
    },
    "D4": {
        "class": A.D4,
        "params": {}
    }
}

colorjitterParams = ["Brightness", "Contrast", "Saturation", "Hue"]
affineParams = ["Translate", "Scale", "Shear"]

imagePath = None
saveDir = None
screenScale = None

def deserialize_config_to_augment_parameters(config_path, augment_parameters):
    """
    Deserialize a configuration file into the augment_parameters variable.

    Args:
        config_path (str): Path to the configuration file.
        augment_parameters (dict): The augment_parameters dictionary to update.

    Returns:
        dict: Updated augment_parameters dictionary.
    """
    import configparser

    # Read the configuration file
    config = configparser.ConfigParser()
    config.read(config_path)

    def UpdateParameter(section, key, value):
        if "Affine" in section or "SafeRotate" in section:
            if value[0] == '-': value = value[1:]
        
        # Convert the value back to its original type (list, float, or int)
        if "," in value:  # List of values
            params[key] = [v.strip() for v in value.split(",")]
            for i in range(len(params[key])):
                _type = float if '.' in params[key][i] else int
                params[key][i] = _type(params[key][i])
        else:  # Single value
            try:
                if '.' in value: params[key] = float(value)
                else: params[key] = int(value)
            except ValueError:
                params[key] = value  # Keep as string if not a number

    # Iterate through each section in the config file
    for section in config.sections():
        if section == "General":
            continue  # Skip the General 
        
        isColor = False
        for color in colorjitterParams:
            if color in section:
                isColor = True
                break
        
        isAffine = False
        for affine in affineParams:
            if affine in section:
                isAffine = True
                break

        if isColor:     augment_section = "ColorJitter"
        elif isAffine:  augment_section = "Affine"
        else:           augment_section = section


        # Check if the section exists in augment_parameters
        if augment_section in augment_parameters:
            params = augment_parameters[augment_section]["params"]

            # Update the parameters from the config file
            for key, value in config[section].items():
                if key == "p": continue  # Skip the probability parameter
                UpdateParameter(section, key, value)

    return augment_parameters

def adjust_list(l, step=0.1):

    if type(l) == int or type(l) == float:
        return l + step
    for i in range(len(l)):
        l[i] = type(l[i])(l[i] + step)
    return l

def CleanValue(val):
    ''' Clean the value up for serialisation into config file '''
    if type(val) == list:
        return str(val).replace("[", "").replace("]", "")
    if type(val) == float:
        return round(val, 4)
    return val

def EqualValueOnly(val):
    ''' Check if all values in the list are equal '''
    if type(val) != list: return val
    
    if len(set(val)) == 1:
        return val[0]
    return None



def GenerateAugmentConfigs(parametersMin):
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    filename = input("Enter augment config file name (optional): ")
    if len(filename) > 0: filename = f"{filename}.conf"
    else: filename = "augment_config.conf"
    config_file_path = os.path.join(saveDir, filename)



    with open(config_file_path, "w") as config_file:
        # General
        config_file.write("[General]\n")
        for augmentName in parametersMin.keys():
            if augmentName == "ColorJitter":
                for param in colorjitterParams:
                    config_file.write(f"{param} = 1\n")
            elif augmentName == "Affine":
                for param in affineParams:
                    config_file.write(f"{param} = 1\n")
            else:
                config_file.write(f"{augmentName} = 1\n")
        # config_file.write("\n")

        # Augmentations
        for augmentName in parametersMin:
            params = parametersMin[augmentName]["params"]
            color = "colorjitter" in augmentName.lower()
            affine = "affine" in augmentName.lower()
            saferotate = "saferotate" in augmentName.lower()

            if not( color or affine ):
                config_file.write(f"\n[{augmentName}]\n")
            
            i = 0
            for param, value, in params.items():
                val = CleanValue(value)

                if affine:  config_file.write(f"\n[Affine{affineParams[i]}]\n")
                if color:   config_file.write(f"\n[{colorjitterParams[i]}]\n")

                if affine or saferotate:
                    if val[0] != '-': val = '-'+val
                
                config_file.write(f"{param} = {val}\n")

                if color or affine:
                    config_file.write("p = 1\n")

                i += 1
            if not( color or affine ): config_file.write("p = 1\n")

    print(f"Configuration saved to {config_file_path}")




    pass

def GenerateFinderConfigs(parametersMin, parametersMax):
    ''' mod = (a(max) - a)/mag(max) '''
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    m_max = float(input("Enter M_max(float): "))

    filename = input("Enter augment finder config file name (optional): ")
    if len(filename) > 0: filename = f"{filename}.conf"
    else: filename = "augment_finder_config.conf"
    config_file_path = os.path.join(saveDir, filename)

    with open(config_file_path, "w") as config_file:
        # General
        if True: # so that I can enclose this :)
            config_file.write("[General]\n")
            config_file.write("name = \n")
            config_file.write("default_augmentation = \n")
            config_file.write("train_config = \n")
            config_file.write("epochs = \n")
            config_file.write("debug_save_dir = \n")
            config_file.write("N_min = \n")
            config_file.write("N_max = \n")
            config_file.write("N_step = \n")
            config_file.write("M_min = 0\n")
            config_file.write(f"M_max = {m_max}\n")
            config_file.write("M_step = \n")
            config_file.write("metric = 2\n")

        def WriteValue(minParams, maxParams, param, m_max, config_file):
            minVal = EqualValueOnly(minParams[param])
            maxVal = EqualValueOnly(maxParams[param])
            if minVal is None: raise(f"Values are not equal. Actually how did you even make it not equal??? Did you touch this script?? Pls don't... minVal({param}): {minVal}")
            if maxVal is None: raise(f"Values are not equal. Actually how did you even make it not equal??? Did you touch this script?? Pls don't... maxVal({param}): {maxVal}")
            mod = (maxVal - minVal)/m_max
            config_file.write(f"{param} = {CleanValue(mod)}\n")

        # Augmentations
        for augmentName in parametersMin:
            minParams = parametersMin[augmentName]["params"]
            maxParams = parametersMax[augmentName]["params"]
            if len(minParams) == 0:
                continue

            if augmentName == "ColorJitter":
                for i, param in enumerate(colorjitterParams):
                    config_file.write(f"\n[{param}]\n")
                    WriteValue(minParams, maxParams, list(minParams.keys())[i], m_max, config_file)
                continue
            elif augmentName == "Affine":
                for i, param in enumerate(affineParams):
                    config_file.write(f"\n[Affine{param}]\n")
                    WriteValue(minParams, maxParams, list(minParams.keys())[i], m_max, config_file)
                continue
            config_file.write(f"\n[{augmentName}]\n")

            for param in minParams.keys():
                WriteValue(minParams, maxParams, param, m_max, config_file)


    print(f"Configuration saved to {config_file_path}")

def GenerateAugmentFinderConfigs(parametersMin, parametersMax):
    '''
    a is the minimum value, the initial value of the parameter
    a(i) = a + a*mag(i)*mod
    Need to find mod 
    a(max) = a + a*mag(max)*mod
    mod = (a(max) - a)/a*mag(max)
    '''

    # Generating augment finder config needs to change n and m, min and max, and the mod.
    # n and m's min and max is given by user input.
    # mod is calculated by the formula above.
    # print("parametersMin111: ", parametersMin)
    GenerateAugmentConfigs(parametersMin)
    GenerateFinderConfigs(parametersMin, parametersMax)

    # Generating augment config needs to change the starting parameter values
    # GenerateAugmentConfigs

    print("Generated configs.")

    pass

def CheckInt(val):
    if type(val) == list:
        return type(val) == int
    return type(val) == int

def main(mainConfig):
    # GenerateAugmentConfigs(augment_parameters)
    # return
    global imagePath
    global saveDir
    global screenScale

    imagePath = mainConfig["AugmentTweaker"]["imagePath"]
    saveDir = mainConfig["AugmentTweaker"]["saveDir"]
    screenScale = float(mainConfig["AugmentTweaker"]["screenScale"])

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    # Load the image
    image = cv2.imread(imagePath)
    
    if image is None:
        print(f"Error: Unable to load image from path '{imagePath}'. Please check the file path.")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (int(image.shape[1] * screenScale), int(image.shape[0] * screenScale)))

    parametersMin = None
    parametersMax = None

    plt.ion()  # Enable interactive mode
    paramNum = 0
    step = 1
    augmentNum = 0
    while True:
        try:
            event = keyboard.read_event()
            if event.event_type == keyboard.KEY_DOWN:
                continue

            keyname = event.name
            # print(f"keyname: {keyname}")
            if keyname == 'esc': break

            augmentName = list(augment_parameters.keys())[augmentNum]

            if keyname == '[':
                augmentNum -= 1
                paramNum = 0
                step = 0.1
                try:
                    if CheckInt(augment_parameters[augmentName]["params"][param]): step = 1
                except: pass
                if augmentNum < 0: augmentNum = len(augment_parameters.keys()) - 1
                augmentName = list(augment_parameters.keys())[augmentNum]
                print("Changed to ", list(augment_parameters.keys())[augmentNum], ", step: ", step)
            elif keyname == ']':
                augmentNum += 1
                paramNum = 0
                step = 0.1
                try:
                    if CheckInt(augment_parameters[augmentName]["params"][param]): step = 1
                except: pass
                if augmentNum >= len(augment_parameters.keys()): augmentNum = 0
                augmentName = list(augment_parameters.keys())[augmentNum]
                print("Changed to ", list(augment_parameters.keys())[augmentNum], ", step: ", step)

            params = list(augment_parameters[augmentName]["params"].keys())
            if len(params) < 1:
                print(f"No parameters for this augment: {augmentName}")
                continue
            param = params[paramNum]
            # There are keyboard controls for the following operations 
            if keyname == 'up':
                augment_parameters[augmentName]["params"][param] = adjust_list(augment_parameters[augmentName]["params"][param], step)
                print(f"keyname: {keyname}, Current {param}: {augment_parameters[augmentName]['params'][param]}")
            elif keyname == 'down':
                augment_parameters[augmentName]["params"][param] = adjust_list(augment_parameters[augmentName]["params"][param], -step)
                print(f"keyname: {keyname}, Current {param}: {augment_parameters[augmentName]['params'][param]}")
            elif keyname == 'left':
                paramNum -= 1
                if paramNum < 0: paramNum = len(augment_parameters[augmentName]["params"].keys()) - 1
                print("Adjusing param: ", list(augment_parameters[augmentName]["params"].keys())[paramNum])
            elif keyname == 'right':
                paramNum += 1
                if paramNum >= len(augment_parameters[augmentName]["params"].keys()): paramNum = 0   
                print("Adjusing param: ", list(augment_parameters[augmentName]["params"].keys())[paramNum])
            elif keyname == 'p':
                print(f"Current parameters:")
                for param, value in augment_parameters[augmentName]["params"].items():
                    print(f"{param}: {value}")
            elif keyname == '-':
                step /= 10
                if CheckInt(augment_parameters[augmentName]["params"][param]) and step < 1: 
                    step = 1
                    print("Step cannot be less than 1 for integer values.")
                print(f"step: {step}")
            elif keyname == '=':
                step *= 10
                print(f"step: {step}")
            elif keyname == 'n':
                # save argument parameters as min
                parametersMin = copy.deepcopy(augment_parameters)
                print("Saved augment parameters as min")
            elif keyname == 'm':
                # save argument parameters as max
                parametersMax = copy.deepcopy(augment_parameters)
                print("Saved augment parameters as max")
            elif keyname == 'b':
                # Generate augment config and finder config and save it
                GenerateAugmentFinderConfigs(parametersMin, parametersMax)
                print("Generated augment and finder configs.")
            elif keyname == 'l':
                config_path = input("Enter the path to the configuration file: ")
                if not os.path.exists(config_path):
                    print(f"Error: Configuration file '{config_path}' does not exist.")
                    continue
                deserialize_config_to_augment_parameters(config_path, augment_parameters)
                print(f"Configuration file '{config_path}' loaded.")
            elif keyname == 's':
                # save config
                print("saving augment config")
                GenerateAugmentConfigs(augment_parameters)
                print("saving augment config done")

            # adjust_parameters(augment_parameters, augmentName, "blur_limit")

            # Apply the augmentation
            augment = augment_parameters[augmentName]["class"](**augment_parameters[augmentName]["params"], p=1, shadow_roi=(0,0,1,1))
            augmented = augment(image=image)
            augmented_image = augmented["image"]
            
                
            combined_image = cv2.hconcat([cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)])
            cv2.imshow("Original and Augmented Image", combined_image)
            cv2.waitKey(1)

            # plt.show()
        except Exception as e:
            import traceback
            print(traceback.format_exc())


    # compose = A.Compose([augment], bbox_params=A.BboxParams(clip = True, format='yolo', label_fields=['category_ids'])) # self.PrepCompose([augment])
    # augmenter = Augmenter()
    # augmentedImage = augmenter.AugmentImage(imagePath, compose)
    
    

    pass







if __name__ == "__main__":
    main()

