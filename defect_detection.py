import cv2
import numpy as np
import os
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def select_test_images():
    """打开文件选择对话框，让用户选择测试图片"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 打开文件选择对话框，允许多选
    file_paths = filedialog.askopenfilenames(
        title='选择测试图片',
        initialdir='train data',  # 默认打开train data目录
        filetypes=[
            ('图片文件', '*.jpg;*.jpeg;*.png;*.JPG;*.JPEG;*.PNG'),
            ('所有文件', '*.*')
        ]
    )
    
    if not file_paths:
        print("未选择任何图片")
        return []
    
    return list(file_paths)


def load_data(data_dir, annot_file):
    """加载图像数据和标签"""
    images = []
    labels = []
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误：目录 {data_dir} 不存在")
        return np.array([]), np.array([])
    
    # 读取标注文件
    try:
        df = pd.read_csv(annot_file)
    except Exception as e:
        print(f"错误：无法读取标注文件 - {str(e)}")
        return np.array([]), np.array([])
    
    # 按文件名分组处理标注
    for filename, group in df.groupby('filename'):
        img_path = os.path.join(data_dir, filename)
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # 调整图像大小为统一尺寸
        img = cv2.resize(img, (224, 224))
        images.append(img)
        
        # 判断图像是否包含损坏区域
        has_damaged = any(
            json.loads(attr)['type'] == 'damaged'
            for attr in group['region_attributes']
        )
        labels.append(1 if has_damaged else 0)
    
    if not images:
        print(f"警告：在 {data_dir} 中没有找到有效的图像")
        return np.array([]), np.array([])
    
    print(f"从 {data_dir} 加载了 {len(images)} 张图像")
    return np.array(images), np.array(labels)


def extract_features(image):
    """提取图像特征"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算HOG特征
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray)
    
    # 计算颜色直方图
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # 计算边缘特征
    edges = cv2.Canny(gray, 100, 200)
    edge_features = cv2.calcHist([edges], [0], None, [32], [0, 256])
    edge_features = cv2.normalize(edge_features, edge_features).flatten()
    
    # 合并特征
    features = np.concatenate([hog_features.flatten(), hist, edge_features])
    return features


def test_images(model, image_paths):
    """测试多张图片"""
    if not image_paths:
        print("错误：没有提供测试图片")
        return
    
    # 计算子图的行列数
    n_images = len(image_paths)
    n_cols = min(5, n_images)  # 每行最多5张图片
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # 创建图形
    plt.figure(figsize=(5*n_cols, 5*n_rows))
    
    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"错误：找不到图片文件 {image_path}")
            continue
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误：无法读取图片文件 {image_path}")
            continue
        
        # 调整图片大小
        img_resized = cv2.resize(img, (224, 224))
        
        # 提取特征并预测
        features = extract_features(img_resized)
        pred = model.predict([features])[0]
        pred_proba = model.predict_proba([features])[0]
        
        # 显示结果
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pred_text = '损坏' if pred == 1 else '完好'
        prob_text = f"置信度: {pred_proba[1]:.2%}"
        plt.title(f"预测: {pred_text}\n{prob_text}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    # 加载所有数据
    print("正在加载数据...")
    annot_path = os.path.join('train data', 'jarlids_annots.csv')
    all_images, all_labels = load_data('train data', annot_path)
    
    if len(all_images) == 0:
        print("错误：数据加载失败，请检查数据目录结构")
        return
    
    # 划分训练集、验证集和测试集
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_images, all_labels, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 提取特征
    print("正在提取特征...")
    train_features = np.array([extract_features(img) for img in X_train])
    val_features = np.array([extract_features(img) for img in X_val])
    test_features = np.array([extract_features(img) for img in X_test])
    
    # 训练模型
    print("正在训练模型...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(train_features, y_train)
    
    # 评估模型
    print("正在评估模型...")
    train_score = model.score(train_features, y_train)
    val_score = model.score(val_features, y_val)
    test_score = model.score(test_features, y_test)
    
    print(f"训练集准确率: {train_score:.4f}")
    print(f"验证集准确率: {val_score:.4f}")
    print(f"测试集准确率: {test_score:.4f}")
    
    # 可视化一些预测结果
    plt.figure(figsize=(15, 5))
    for i in range(min(5, len(X_test))):
        plt.subplot(1, 5, i+1)
        img = X_test[i]
        pred = model.predict([extract_features(img)])[0]
        true_label = y_test[i]
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pred_text = '损坏' if pred == 1 else '完好'
        true_text = '损坏' if true_label == 1 else '完好'
        plt.title(f"预测: {pred_text}\n实际: {true_text}")
        plt.axis('off')
    plt.show()
    
    # 让用户选择测试图片
    print("\n请选择要测试的图片...")
    test_images_paths = select_test_images()
    
    if test_images_paths:
        print(f"选择了 {len(test_images_paths)} 张图片进行测试")
        test_images(model, test_images_paths)
    else:
        print("未选择任何图片，程序结束")


if __name__ == "__main__":
    main() 