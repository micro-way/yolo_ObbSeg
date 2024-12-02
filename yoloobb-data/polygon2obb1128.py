import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_polygon_and_rbox_on_image(image, polygons_data, save_image_path):
    """在同一张图像上绘制所有多边形和旋转框"""
    img = image.copy()

    for object_type, polygon, rbox in polygons_data:
        # 将多边形的顶点转换为适合 OpenCV 画图的格式
        polygon_np = np.array(polygon, dtype=np.int32)
        polygon_np = polygon_np.reshape((-1, 1, 2))

        # 绘制多边形
        cv2.polylines(img, [polygon_np], isClosed=True, color=(0, 255, 0), thickness=2)

        # 从 rbox 数据中提取旋转框的参数
        (cx, cy, w, h, angle) = rbox

        # 通过 cv2.boxPoints 从旋转框获取 4 个顶点
        rect = ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # 将 np.int0 改为 np.intp

        # 绘制旋转框
        cv2.drawContours(img, [box], 0, (255, 0, 0), 2)

        # 显示物品类型标签
        cv2.putText(img, f'Object {object_type}', (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 显示图像
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.title('Polygons and Rotated Bounding Boxes')
    #plt.show()

    # 保存最终绘制的图像
    cv2.imwrite(save_image_path, img)


def polygon_to_rbox(polygon):
    """将多边形顶点转换为旋转包围框"""
    polygon = np.array(polygon, dtype=np.float32)
    rect = cv2.minAreaRect(polygon)
    (cx, cy), (w, h), angle = rect

    if w < h:
        w, h = h, w
        angle += 90
    return cx, cy, w, h, angle


def parse_txt_file(txt_file, image_shape):
    """解析txt文件并返回多边形的物品类型和坐标"""
    data = []
    with open(txt_file, 'r') as file:
        for line in file:
            elements = line.strip().split()
            object_type = int(elements[0])  # 物品类型
            relative_coords = [float(coord) for coord in elements[1:]]  # 相对坐标
            absolute_coords = []
            # 将相对坐标转换为像素坐标
            for i in range(0, len(relative_coords), 2):
                x = int(relative_coords[i] * image_shape[1])  # 相对宽度转为像素
                y = int(relative_coords[i + 1] * image_shape[0])  # 相对高度转为像素
                absolute_coords.append((x, y))
            data.append((object_type, absolute_coords))
    return data


'''def save_rbox_data_to_txt(rbox_data, output_file):
    """将旋转框数据保存到txt文件"""
    with open(output_file, 'w') as file:
        for object_type, rbox in rbox_data:
            cx, cy, w, h, angle = rbox
            # 确保数值使用适当的小数点精度
            file.write(f'{object_type} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {angle:.6f}\n')'''


def save_rbox_vertices_to_txt(rbox_data, output_file):
    """将旋转框数据保存到txt文件，包括物品类型和四点坐标的相对位置"""
    img_height, img_width = 1024, 1024  # 获取图像的高度和宽度
    with open(output_file, 'w') as file:
        for object_type, rbox in rbox_data:
            (cx, cy, w, h, angle) = rbox

            # 计算旋转框的四个顶点
            rect = ((cx, cy), (w, h), angle)
            box = cv2.boxPoints(rect)  # 获取旋转框的四个顶点
            box = np.intp(box)  # 将顶点坐标转换为整数

            # 将顶点坐标转换为相对位置（相对图像宽度和高度）
            relative_coords = []
            for (x, y) in box:
                rel_x = x / img_width  # x 相对图像宽度
                rel_y = y / img_height  # y 相对图像高度
                rel_x = max(0.000001, min(rel_x, 1.0))
                rel_y = max(0.000001, min(rel_y, 1.0))
                relative_coords.append(f'{rel_x:.6f} {rel_y:.6f}')

            # 写入物品类型和相对坐标
            file.write(f'{object_type} {" ".join(relative_coords)}\n')


def process_image_with_txt(image_file, txt_file, output_txt_file, save_image_path):
    """读取图像和txt文件，绘制多边形和旋转框，并保存旋转框数据"""
    # 读取图像
    #if not isinstance(image_file, str) or not os.path.exists(image_file):
    #   print(f"Invalid file path: {image_file}")
    image_file = image_file[0]
    txt_file=txt_file[0]
    image = cv2.imread(image_file)
    image_shape = image.shape

    # 解析txt文件
    polygons_data = parse_txt_file(txt_file, image_shape)

    # 生成旋转框数据
    rbox_data = []
    for object_type, polygon in polygons_data:
        # 计算旋转包围框
        rbox = polygon_to_rbox(polygon)
        rbox_data.append((object_type, rbox))

    # 保存旋转框数据到txt文件
    save_rbox_vertices_to_txt(rbox_data, output_txt_file)

    # 在图像上绘制所有的多边形和旋转框
    draw_polygon_and_rbox_on_image(image, [(obj_type, poly, rbox) for (obj_type, poly), (_, rbox) in
                                           zip(polygons_data, rbox_data)], save_image_path)


# 使用示例
#image_file = 'E:/wheat/ultralytics-main/temp/20240509101652_6.jpg'  # 替换为实际的图像文件路径
#txt_file = 'E:/wheat/ultralytics-main/temp/20240509101652_6.txt'  # 替换为实际的txt文件路径
#output_txt_file = 'rbox_data_output.txt'  # 输出的旋转框数据txt文件
#save_image_path = 'output_image.jpg'  # 保存带有旋转框的图像
image_folder = r'C:\Users\zjzcw\Desktop\yoloobb-data\InstSeg3474yolo622\test\images'
txt_folder = r'C:\Users\zjzcw\Desktop\yoloobb-data\InstSeg3474yolo622\test\labels'
output_txt_folder = r'C:\Users\zjzcw\Desktop\yoloobb-data\InstSeg3474yolo622\test\obb_labels'
output_image_folder = r'C:\Users\zjzcw\Desktop\yoloobb-data\InstSeg3474yolo622\test\obb_images'

for image_name in os.listdir(image_folder):
    image_paths = []
    txt_paths = []
    if image_name.endswith('.jpg'):
        # 获取文件名（去掉扩展名）
        file_name_no_ext = os.path.splitext(image_name)[0]

        # 在txt_folder中查找同名txt文件
        for txt_file in os.listdir(txt_folder):
            if txt_file.endswith('.txt'):
                txt_file_no_ext = os.path.splitext(txt_file)[0]

                # 如果找到同名文件，将路径分别添加到jpg_paths和txt_paths
                if file_name_no_ext == txt_file_no_ext:
                    image_paths.append(os.path.join(image_folder, image_name))
                    txt_paths.append(os.path.join(txt_folder, txt_file))
                    output_txt_path = os.path.join(output_txt_folder, txt_file)
                    output_image_path = os.path.join(output_image_folder, image_name)
                    process_image_with_txt(image_paths, txt_paths, output_txt_path, output_image_path)

    #return jpg_paths, txt_paths

# 调用处理函数
#process_image_with_txt(image_file, txt_file, output_txt_file, save_image_path)
