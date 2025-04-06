''' This script is responsible for generating a folder of generating test augmented images and labels with the largest and smallest magnitude multiplier for augmentation. '''



def main():
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../AutoAugment'))) # AutoAugment's directory
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # src's directory
    from RandAugmentFinder import RandAugmentFinder
    from Augmenter import Augmenter
    import RandAugmentGenerator
    
    a = RandAugmentFinder("augmentFinderConfigs/skyfusion_all_non_spatial_finder.conf")

    mag = [a.M_min, a.M_max]

    for m in mag:
        RandAugmentGenerator.ApplyRandAugmentMagnitude(a.default_augmentation,a.augmentFinderConfig, m)
        augmenter = Augmenter()
        augmenter.augmentList = RandAugmentGenerator.globalAugmentationList
        
        augmenter.TestAugment("C:/Users/TeohJ/Desktop/Data Prep/datasets/trial/val/MA_img_24.jpg", f"C:/Users/TeohJ/Desktop/FlexiVisionSystem/AutoTrainer/ExtremeAugmentDebug/img_{m}m.jpg")





if __name__ == "__main__":
    main()







