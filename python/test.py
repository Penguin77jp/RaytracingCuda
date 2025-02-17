from raytracing import *
import matplotlib.pyplot as plt
import numpy as np

# 光学系の設定
def create_optical_system():
    path = ImagingPath()
    path.label = "2枚レンズシステム with 回折"
    path.append(Space(d=50))      # 物体から第1レンズまで
    path.append(Lens(f=50, diameter=25, label="レンズ1"))
    path.append(Space(d=75))      # 第1レンズから絞りまで
    path.append(Aperture(diameter=10, label="絞り"))
    path.append(Space(d=75))      # 絞りから第2レンズまで
    path.append(Lens(f=70, diameter=30, label="レンズ2"))
    path.append(Space(d=100))     # 第2レンズから像面まで
    return path

# 回折効果を考慮した光線生成
def generate_diffracted_rays(aperture_diameter, num_rays=100):
    rays = []
    for _ in range(num_rays):
        # 絞り面でランダムな位置をサンプリング
        x = np.random.uniform(-aperture_diameter / 2, aperture_diameter / 2)
        y = np.random.uniform(-aperture_diameter / 2, aperture_diameter / 2)
        
        # 光線の方向（ここでは簡単化して直進）
        angle_x = np.random.uniform(-0.01, 0.01)  # 小さな角度でランダム化
        angle_y = np.random.uniform(-0.01, 0.01)
        
        rays.append(Ray(y=y, theta=angle_x))
    return rays

# 可視化メイン処理
def visualize_system():
    # 光学系の作成
    system = create_optical_system()
    
    # 光線生成と追跡
    aperture = next(c for c in system.elements if isinstance(c, Aperture))
    aperture_diameter = aperture.apertureDiameter  # 正しい属性名を使用
    
    rays = generate_diffracted_rays(aperture_diameter, num_rays=50)
    
    # 光学系の表示（エラー修正：ax引数を削除）
    system.display()
    
if __name__ == "__main__":
    visualize_system()
