''' A script that takes in a folder of images and text file containing the bounding box of the image. Then present the image with bounding box overlaying it '''
import os
import cv2

def draw_bounding_boxes(image, bboxes):
    h, w, _ = image.shape
    for bbox in bboxes:
        obj_id, x_center, y_center, width, height = bbox
        x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(obj_id), (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    return image

def read_bounding_boxes(file_path):
    with open(file_path, 'r') as file:
        bboxes = [line.strip().split() for line in file.readlines()]
    return [[int(bbox[0])] + list(map(float, bbox[1:])) for bbox in bboxes]

def main(mainConfig):
    folderPath = mainConfig["ViewImages"]["folderPath"]

    image_files = [f for f in os.listdir(folderPath) if f.endswith('.jpg') or f.endswith('.png')]
    print("total images: ", len(image_files))
    current_image_index = 0

    def show_image(index):
        image_file = image_files[index]
        image_path = os.path.join(folderPath, image_file)
        bbox_path = image_path.replace(".jpg", ".txt").replace(".png", ".txt")

        if os.path.exists(bbox_path):
            print("Processing ", image_file)
            image = cv2.imread(image_path)
            bboxes = read_bounding_boxes(bbox_path)
            image = draw_bounding_boxes(image, bboxes)
            imageScale = 1.6
            image = cv2.resize(image, (int(imageScale * (image.shape[1] // 4)), int(imageScale * (image.shape[0] // 4))))
            cv2.imshow('View Images', image)

    def on_key(key):
        nonlocal current_image_index
        if key == 'right':
            current_image_index = (current_image_index + 1) % len(image_files)
        elif key == 'left':
            current_image_index = (current_image_index - 1) % len(image_files)
        show_image(current_image_index)

    show_image(current_image_index)
    # cv2.setMouseCallback('Image with Bounding Boxes', on_key)
    while True:
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to exit
            break
        elif key == ord('a'):  # 'A' key
            on_key('left')
        elif key == ord('d'):  # 'D' key
            on_key('right')
        elif cv2.getWindowProperty('View Images', cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




