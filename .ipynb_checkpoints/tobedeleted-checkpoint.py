from ultralytics.models.yolo.classify import ClassificationTrainer

if __name__ == '__main__':
    args = dict(model="yolo11n-cls.pt", data="mnist160", batch = 15, epochs=3, imgsz = 64, augment = True, auto_augment= "randaugment")
    trainer = ClassificationTrainer(overrides=args)
    trainer.train()
