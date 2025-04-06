''' This script is responsible for running the augmenter with the given configuration files, generating augmented images and labels. '''





def main(mainConfig):
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../AutoAugment'))) # AutoAugment's directory
    from Augmenter import Augmenter

    augmenter = Augmenter()
    augmenter.PrepCompose(mainConfig["DoAugment"]["augmentPath"])

    paths = [path.strip() for path in mainConfig["DoAugment"]["sourcePaths"].split(",")]
    augmenter.AugmentAndSave(paths, mainConfig["DoAugment"]["savePath"])


if __name__ == "__main__":
    main()







