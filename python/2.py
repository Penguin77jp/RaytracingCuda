import numpy as np
import matplotlib.pyplot as plt

# --- ステップ2: 3D（x, y, z）での厚レンズ射影変換 ---
# パラメータ（単位: mm）
f = 50.0  # 焦点距離

# 3D版の4×4 射影変換行列
# 入力ベクトルは [x, y, z, 1]^T とする。
# 行列は、x, y 成分は f 倍し、w 成分は (z - f) とする。
M_3D = np.array([
    [f,   0, 0,  0],
    [0,   f, 0,  0],
    [0,   0, 0,  0],   # 画像面 z = 0 に射影するため
    [0,   0, 1, -f]
])

print("【ステップ2】3D厚レンズ変換の4×4行列:")
print(M_3D)

# 変換を適用する関数（3D版）
def apply_transform_3D(x, y, z, M):
    vec = np.array([x, y, z, 1])
    transformed = M @ vec
    w = transformed[3]
    if np.abs(w) < 1e-8:
        return None
    result = transformed / w
    # 結果は [x', y', z', 1]。画像面では z' ≈ 0 となる。
    return result

# テスト：異なる物体距離と物体面上の (x, y) のグリッド点について変換を適用
object_distance = 150.0  # 固定の物体面までの距離 (mm)
grid_range = np.linspace(-10, 10, 5)
X, Y = np.meshgrid(grid_range, grid_range)
X_flat = X.flatten()
Y_flat = Y.flatten()

projected_points = []
for x, y in zip(X_flat, Y_flat):
    res = apply_transform_3D(x, y, object_distance, M_3D)
    if res is not None:
        projected_points.append([res[0], res[1]])
projected_points = np.array(projected_points)

# 理論的には、薄レンズ公式により
# x_img = f*x/(object_distance - f), y_img = f*y/(object_distance - f)
scale = f / (object_distance - f)
theoretical_x = scale * X_flat
theoretical_y = scale * Y_flat

plt.figure(figsize=(8, 6))
plt.plot(projected_points[:,0], projected_points[:,1], 'ro', label='射影後の点')
plt.plot(theoretical_x, theoretical_y, 'b+', label='理論値')
plt.xlabel("画像面 x (mm)")
plt.ylabel("画像面 y (mm)")
plt.title("【ステップ2】3D 4×4射影変換による画像面上の点")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
