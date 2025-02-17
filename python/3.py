import numpy as np
import matplotlib.pyplot as plt

# --- ステップ3: 複数枚レンズによるシステムのシミュレーション ---
# 自由空間伝搬行列（4×4）
def T(d):
    # d: 伝搬距離 (mm)
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, d],
        [0, 0, 0, 1]
    ])

# 薄レンズ変換行列（4×4）
def lens_matrix(f):
    # f: レンズの焦点距離 (mm)
    # この行列は、入力 [x, y, z, 1]^T に対し、射影除算後に
    # x_img = f*x/(z - f), y_img = f*y/(z - f) となる（画像面は z = 0 とする）
    return np.array([
        [f,  0, 0, 0],
        [0,  f, 0, 0],
        [0,  0, 0, 0],
        [0,  0, 1, -f]
    ])

# システムパラメータ例
# 例として、2枚のレンズシステムを考える。
f1 = 80.0   # 第1レンズの焦点距離 (mm)
f2 = 120.0  # 第2レンズの焦点距離 (mm)

# 自由空間伝搬距離（レンズ前後、レンズ間）
d1 = 100.0  # 物体面から第1レンズまで
d2 = 50.0   # 第1レンズから第2レンズまで
d3 = 40.0   # 第2レンズから画像面まで

# 各要素の変換行列を計算
M_lens1 = lens_matrix(f1)
M_lens2 = lens_matrix(f2)

T1 = T(d1)  # 物体面から第1レンズまで
T2 = T(d2)  # 第1レンズから第2レンズまで
T3 = T(d3)  # 第2レンズから画像面まで

# 全体の変換行列（右側の行列が最初に作用することに注意）
# M_total = T3 * M_lens2 * T2 * M_lens1 * T1
M_total = T3 @ M_lens2 @ T2 @ M_lens1 @ T1

print("【ステップ3】複数枚レンズシステムの全体変換行列:")
print(M_total)

# 変換を適用する関数（3D版、全システム）
def apply_total_transform(x, y, z, M):
    vec = np.array([x, y, z, 1])
    transformed = M @ vec
    w = transformed[3]
    if np.abs(w) < 1e-8:
        return None
    result = transformed / w
    # 結果は [x_img, y_img, z_img, 1]。画像面では z_img ≈ 0 とする。
    return result

# テスト：物体面上のグリッド点（例えば、物体面 z = 0 として T1 により d1 だけ進んだ状態を入力）
# ※ここでは物体面での座標を (x, y) として用い、システム全体の作用を確認します。
grid_range = np.linspace(-10, 10, 7)
X, Y = np.meshgrid(grid_range, grid_range)
X_flat = X.flatten()
Y_flat = Y.flatten()

projected_points = []
for x, y in zip(X_flat, Y_flat):
    # 入力点の z 座標は 0（物体面）ですが、T1 により d1 だけ進むので実質 z = 0 + d1
    res = apply_total_transform(x, y, 0, M_total)
    if res is not None:
        projected_points.append([res[0], res[1]])
projected_points = np.array(projected_points)

# ここでは理論的な値は簡単には求めにくいので、単に全体変換による画像面上の分布を可視化
plt.figure(figsize=(8, 6))
plt.plot(projected_points[:,0], projected_points[:,1], 'ro', label='画像面上の点')
plt.xlabel("画像面 x (mm)")
plt.ylabel("画像面 y (mm)")
plt.title("【ステップ3】複数枚レンズシステムによる画像面射影")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
