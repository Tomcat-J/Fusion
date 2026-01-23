import os
import shutil

def move_images(source_dir, target_dir, ratio=0.8):
    # 获取源文件夹中的所有文件名，并按照名字排序
    files = sorted([f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))])

    # 计算需要移动的文件数量
    num_files_to_move = int(len(files) * ratio)

    # 创建目标文件夹，如果不存在的话
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 移动文件
    for i in range(num_files_to_move):
        shutil.move(os.path.join(source_dir, files[i]), os.path.join(target_dir, files[i]))

# 使用示例
source_dir = r'C:\Users\wyq\Desktop\2221\MIA'  # 源文件夹路径
target_dir = r'C:\Users\wyq\Desktop\lung\train\MIA'  # 目标文件夹路径
move_images(source_dir, target_dir)