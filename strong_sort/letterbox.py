import cv2


def cv2_letterbox_image(image, expected_size=(128, 256)):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img


if __name__ == "__main__":
    image = cv2.imread(r"E:\VisDrone-MOT\ReID\query\2_1.jpg")
    expect_size = (128, 256)
    new_img = cv2_letterbox_image(image, expect_size)

    new_img_2 = cv2.resize(image, expect_size)

    cv2.imwrite(r"E:\VisDrone-MOT\ReID\query\899899889.jpg", new_img)
    cv2.imwrite(r"E:\VisDrone-MOT\ReID\query\222222222222.jpg", new_img_2)



