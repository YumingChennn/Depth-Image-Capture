import numpy as np
import cv2
from matplotlib import pyplot as plt


def getBackgoround():
    # 读取原始图像
    original_image = cv2.imread("xaycNDmeyxpHDrPqU6LmaD.jpg")

    # 将图像大小调整为12080x720
    resized_image = cv2.resize(original_image, (1280, 720))

    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return gray_image

imgL = cv2.imread("yumingLeft.png", 0)
imgR = cv2.imread("yumingRight.png", 0)


ratio = 0.5
# Resizing images
resized_imgL = cv2.resize(imgL, (int(1280*ratio), int(720*ratio)))
resized_imgR = cv2.resize(imgR, (int(1280*ratio), int(720*ratio)))



# Create a StereoBM object
stereo = cv2.StereoBM_create()

#disparity = stereo.compute(imgL, imgR)
# Set parameters individually
stereo.setNumDisparities(80)   # Number of disparities. Adjust as needed.
stereo.setBlockSize(11)         # Block size. Adjust as needed.


# Compute disparity map
disparity = stereo.compute(imgL, imgR)



# Convert disparity to uint8
disparity_uint8 = cv2.convertScaleAbs(disparity)
disparity_uint16 = disparity.astype(np.int16)

# Determine center pixel coordinates
center_x = disparity.shape[1] // 2
center_y = disparity.shape[0] // 2

# Get center pixel value
center_pixel_value = disparity_uint16[center_y, center_x]

# Define threshold for filtering out pixels with large differences
threshold = 179

# Filter out pixels with large differences from center pixel
for y in range(disparity.shape[0]):
    for x in range(disparity.shape[1]):
        diff = abs(int(disparity_uint16[y, x]) - int(center_pixel_value))
        diff_clipped = np.clip(diff, 0, 255)  # 将差异限制在 uint8 的范围内
        if diff_clipped > threshold:
            disparity_uint16[y, x] = 0

disparity_uint16 = disparity_uint16.astype(np.uint8)

disparity_uint16 = cv2.medianBlur (disparity_uint16,7)



# 提取最大的白色区域
# 二值化图像，将白色区域设为前景（255），黑色区域设为背景（0）
_, binary_image = cv2.threshold(disparity_uint16, 0, 255, cv2.THRESH_BINARY)

# 进行连通组件标记
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

# 找到最大的白色区域
max_label = 1  # 默认情况下，标签0是背景，所以我们从标签1开始
max_area = stats[max_label, cv2.CC_STAT_AREA]
for label in range(2, num_labels):
    area = stats[label, cv2.CC_STAT_AREA]
    if area > max_area:
        max_label = label
        max_area = area

# 创建一个只包含最大白色区域的二值图像
largest_white_region = (labels == max_label).astype(np.uint8) * 255

for i in range(1,20):
    largest_white_region = cv2.medianBlur (largest_white_region,19)


# 将最大白色区域应用于原始图像
result_img = imgL.copy()
# 重新调整 largest_white_region 的大小以匹配 result_img 的大小
largest_white_region = cv2.resize(largest_white_region, (result_img.shape[1], result_img.shape[0]))
largest_black_region = cv2.bitwise_not(largest_white_region)
result_img[largest_black_region == 255] = 0

# 获取 result_img 的大小
#height, width = result_img.shape
#print("result_img 的大小：", width, "x", height)

gray_image = getBackgoround()


result_image = cv2.bitwise_and(gray_image, largest_black_region)
result_image = cv2.bitwise_or(result_img, result_image)


# Show the processed disparity map
plt.imshow(result_image, "gray")
plt.show()
