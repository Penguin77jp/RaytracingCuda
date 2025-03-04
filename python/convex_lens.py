#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_lens_interface(ax, x0, radius, diameter, n_points=200, color='blue'):
    """
    指定された頂点位置 x0、曲率半径 radius、レンズ径 diameter から
    魅力的な球面（レンズ面）の座標を算出し、ax上にプロットする関数です。

    引数:
       ax: matplotlibのAxesオブジェクト
       x0: 面の頂点 (光軸との交点) のx座標 
       radius: 球面の曲率半径（正なら凸面、負なら凹面）
       diameter: レンズの直径
       n_points: 曲線描画に用いる分割点数（デフォルト200）
       color: 曲線の色（デフォルトはblue）
    """
    # y軸方向を–diameter/2〜+diameter/2の範囲で分割
    y = np.linspace(-diameter/2, diameter/2, n_points)
    # 計算時の浮動小数点誤差対策として、平方根内部が負にならないようclip
    inside_sqrt = np.clip(radius**2 - y**2, 0, None)
    if radius >= 0:
        # 正の半径の場合: 球の中心は頂点より右側にあるので、x座標は
        # x(y) = x0 + R - √(R² - y²)
        x = x0 + radius - np.sqrt(inside_sqrt)
    else:
        # 負の半径の場合: 球の中心は頂点より左側にあるので、x(y) = x0 + R + √(R² - y²)
        x = x0 + radius + np.sqrt(inside_sqrt)
    ax.plot(x, y, color=color, linewidth=2)

# JSON形式の辞書によるレンズシステムの定義
lens_system = {}
# ここではconvex lensの2面を定義（第一面：空気→レンズ、第二面：レンズ→空気）
lens_system['lens'] = []
lens_system['lens'].append({
    'thickness' : 0.5,       # 第一面の頂点位置から面までのずれ
    'radius' : 0.5,          # 第一面の曲率半径（正：凸面）
    'diameter' : 0.84,        # レンズ径
    'refractive_index' : 1.5,
})
lens_system['lens'].append({
    'thickness' : 0.5,       # 面間隔（次の面へ進むための距離）
    'radius' : -0.5,         # 第二面の曲率半径（負：凹面）
    'diameter' : 0.84,
    'refractive_index' : 1.0,
})

# プロットの準備
fig, ax = plt.subplots(figsize=(8, 4))
position_offset = 0.0  # 各面の頂点（光軸との交点）のx座標

# 各媒質境界面（レンズ面）をプロット
for lens in lens_system['lens']:
    thickness = lens['thickness']
    radius = lens['radius']
    diameter = lens['diameter']
    refractive_index = lens['refractive_index']
    
    # 光軸との交点（vertex）を黒い点でプロット
    ax.scatter(position_offset, 0, color='black', zorder=5)
    
    # 球面（レンズ面）のプロット
    plot_lens_interface(ax, position_offset, radius, diameter, color='blue')
    
    # 厚み分だけ次の面へx方向にシフト
    position_offset += thickness

# グラフの見た目を調整
ax.set_aspect('equal')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Convex Lens Profile')
ax.grid(True)


# レンズ数などの情報もJSONに追加し、必要に応じてファイルに出力可能
lens_system['num'] = len(lens_system['lens'])

# https://www.optics-words.com/kikakogaku/focal_length.html
refractive_index = lens_system['lens'][0]['refractive_index']
radius0 = lens_system['lens'][0]['radius']
radius1 = lens_system['lens'][1]['radius']
thickness = lens_system['lens'][0]['thickness']
focal_length_f = (refractive_index - 1.0) * (1.0/radius0 - 1.0/radius1) + thickness / refractive_index * (refractive_index - 1.0)**2 / radius0 / radius1
focal_length = 1.0 / focal_length_f
print('focal length', focal_length)

# principal planes
## front
H = - thickness * (refractive_index - 1.0) / refractive_index / radius0
## back
H_DASH = thickness * (1.0 - refractive_index) / refractive_index / radius1

lens_system['focal_length'] = focal_length
lens_system['fonrt_principal_planes'] = H
lens_system['back_principal_planes'] = H_DASH


import json
with open('../assets/lens_system_test.json', 'w') as f:
    json.dump(lens_system, f, indent=4)


plt.show()
