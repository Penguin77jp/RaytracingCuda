import numpy as np
import matplotlib.pyplot as plt

# --- ステップ1: 2D（x–z平面）における厚レンズ射影変換の4×4行列 ---
# パラメータ（単位: mm）
f = 50.0  # 焦点距離（ここでは f = 50mm とする）

# 4×4 射影変換行列（2D版）
# 入力ベクトルは [x, 0, z, 1]^T とする。
# この行列は、変換後の同次座標で
#   x' = f * x,   w' = z - f
# として、射影除算後に x_img = (f * x)/(z - f) となる（薄レンズの場合の拡大率 m = f/(z-f)）。
M_2D = np.array([
    [f,   0, 0,  0],
    [0,   1, 0,  0],   # y 成分はそのまま（2Dでは常に 0）
    [0,   0, 0,  0],   # 出力 z を 0（画像面）に射影するための行
    [0,   0, 1, -f]
])

print("【ステップ1】2D厚レンズ変換の4×4行列:")
print(M_2D)

# 変換を適用する関数（2D版）
def apply_transform_2D(x, z, M):
    # 入力は [x, 0, z, 1]（2Dなので y は 0）
    vec = np.array([x, 0, z, 1])
    transformed = M @ vec
    w = transformed[3]
    if np.abs(w) < 1e-8:
        return None
    result = transformed / w  # 射影除算
    # 結果は [x', y', z', 1] となる。ここでは y' は 0, z' は 0 となるはず。
    return result

# テスト：いくつかの入力 (x, z) に対して変換結果を求める
# object_distances: 物体までの距離 (z)
object_distances = np.linspace(60, 200, 5)  # f より大きい値（実像が得られる）
x_values = np.linspace(-10, 10, 5)            # 物体面での高さ

results = []  # 各 z における変換結果を保存
for z in object_distances:
    x_img_list = []
    for x in x_values:
        res = apply_transform_2D(x, z, M_2D)
        if res is not None:
            x_img_list.append(res[0])  # 変換後の x 座標
        else:
            x_img_list.append(np.nan)
    results.append((z, x_img_list))

# プロット：物体面での x と、画像面での x_img の関係（薄レンズの公式 x_img = f*x/(z-f)）
plt.figure(figsize=(8, 6))
for z, x_img_list in results:
    # 理論値（薄レンズ公式）
    x_theoretical = f * x_values / (z - f)
    plt.plot(x_values, x_img_list, 'o-', label=f'物体距離 z = {z:.0f} mm')
    plt.plot(x_values, x_theoretical, 'k--', alpha=0.5)
    
plt.xlabel("物体面での高さ x (mm)")
plt.ylabel("画像面での高さ x_img (mm)")
plt.title("【ステップ1】2D 4×4射影変換による厚レンズ変換")
plt.legend()
plt.grid(True)
plt.show()
