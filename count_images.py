import os

def count_images_in_folder(folder_path):
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    image_count = 0
    
    # 遍历文件夹及其子文件夹
    for root, _, files in os.walk(folder_path):
        for file in files:
            # 将文件名转换为小写，以便不区分大小写地匹配扩展名
            if file.lower().endswith(image_extensions):
                image_count += 1
    return image_count

def count_images_in_dataset(base_data_path):
   
    print(f"--- 统计数据集 '{base_data_path}' 中的图片数量 ---")
    
    total_images = 0
    
    # 假设你的数据集结构是 base_data_path/category_name/image.jpg
    # 所以需要遍历 base_data_path 下的每个子文件夹（即类别文件夹）
    
    # 获取所有类别文件夹的列表
    category_folders = [d for d in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, d))]
    
    if not category_folders:
        print(f"在 '{base_data_path}' 中没有找到任何子文件夹（类别）。请检查路径。")
        return

    for category_folder in sorted(category_folders): # 按字母顺序排序，方便查看
        category_path = os.path.join(base_data_path, category_folder)
        count = count_images_in_folder(category_path)
        print(f"类别 '{category_folder}': {count} 张图片")
        total_images += count
        
    print(f"\n--- 数据集总图片数量: {total_images} 张 ---")

if __name__ == "__main__":
    # 示例: 假设你的图片在 dog_project/data/active, dog_project/data/eating 等
    dataset_root = 'data' 
    
    # 确保路径存在
    if not os.path.exists(dataset_root):
        print(f"错误: 指定的数据集根目录 '{dataset_root}' 不存在。请检查路径。")
    else:
        count_images_in_dataset(dataset_root)
        