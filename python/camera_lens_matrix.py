import numpy as np
import matplotlib.pyplot as plt

# 2x2 行列の掛け算（行列積）
def mat_mult(A, B):
    return np.dot(A, B)

# レンズ面での屈折の ABCD 行列
# 入射媒質の屈折率 n1 から出射媒質の屈折率 n2 への球面面（曲率半径 R）での屈折行列
def refraction_matrix(n1, n2, R):
    # 近軸近似における屈折行列
    # M = [ [1, 0],
    #       [ (n1 - n2)/(R * n2), n1/n2 ] ]
    return np.array([[1.0, 0.0],
                     [ (n1 - n2)/(R * n2), n1/n2 ]])

# 媒質中の距離 d の伝搬行列
def translation_matrix(d):
    # M = [ [1, d],
    #       [0, 1] ]
    return np.array([[1.0, d],
                     [0.0, 1.0]])

def camera_lens(ax, n_air, n_glass, R1, R2, d, z_lim):
    # --- 各要素の ABCD 行列 ---
    # 第1面：空気 -> ガラス
    M1 = refraction_matrix(n_air, n_glass, R1)
    # レンズ内の伝搬（厚さ d のガラス中）
    Md = translation_matrix(d)
    # 第2面：ガラス -> 空気
    M2 = refraction_matrix(n_glass, n_air, R2)

    # レンズ全体の ABCD 行列（順番に作用するので右から掛ける： M_total = M2 * Md * M1 ）
    M_total = mat_mult(M2, mat_mult(Md, M1))

    print("厚レンズシステムの ABCD 行列（近軸近似）:")
    print(M_total)

    # --- 焦点距離の計算 ---
    # 入射光線：平行光線 (角度 theta=0) からの入射の場合、出力は (x_out, theta_out) = (A*x_in, C*x_in)
    # 平行光線が焦点で収束するため、自由空間伝搬（translation）を考えると、
    #  x_out + L * theta_out = 0  → L = - x_out/theta_out = -A/C  (C ≠ 0)
    A, B, C, D = M_total[0,0], M_total[0,1], M_total[1,0], M_total[1,1]
    if np.abs(C) > 1e-8:
        focal_length = -A / C
    else:
        focal_length = np.inf

    print("\n計算された焦点距離 f' = {:.2f} mm".format(focal_length))

    # --- 可視化のための光線追跡 ---
    # レンズ直後の状態（M_total により変換された状態）から自由空間伝搬で焦点面まで光線が収束する様子を描画する
    # 光線は (x, theta) の形で表現（x: 高さ, theta: 角度 (ラジアン)）
    # ここでは、入力光線としてレンズ前において高さ x_in をいろいろ変化させ、角度は 0 とする
    num_rays = 7
    x_inputs = np.linspace(-5, 5, num_rays)  # 入射光線の高さ（mm）
    theta_inputs = np.zeros(num_rays)         # 平行光線（角度0）
    rays_in = np.vstack([x_inputs, theta_inputs])  # 2×N 行列

    # レンズ全体の行列 M_total を適用（入力がレンズ直前とみなす）
    rays_after = M_total @ rays_in  # (x_out, theta_out) = (A*x_in, C*x_in) となるはず
    # カメラレンズの形状をプロット
    # レンズは z = -d（レンズ入口）から z = 0（レンズ出口）に存在すると仮定
    aperture = 6.0  # レンズの半径（mm）
    num_points = 100
    x_arc = np.linspace(-aperture, aperture, num_points)

    # 前面（入口面）の曲面：小角近似で sag = x^2/(2*R1)
    z_front = -d + (x_arc**2) / (2 * R1)
    # 後面（出口面）の曲面：R2 は負なので sag = x^2/(2*R2)（実質的に減少）
    z_back = 0 + (x_arc**2) / (2 * R2)
    print('z_front:', z_front)
    print('z_back:', z_back)

    # 多角形を作成（出口面の点から入口面の点へ逆順に連結）
    z_poly = np.concatenate((z_back, z_front[::-1]))
    x_poly = np.concatenate((x_arc, x_arc[::-1]))

    ax.fill(z_poly, x_poly, color='gray', alpha=0.3, label='Camera Lens')
    # 焦点面（z = focal_length）で各光線がどこに交わるか計算
    # 自由空間伝搬 T(L) = [[1, L], [0, 1]] で、光線の高さは x(L) = x_out + L*theta_out
    # L = focal_length で x(focal_length) = x_out + focal_length * theta_out
    x_focal = rays_after[0, :] + focal_length * rays_after[1, :]

    # 可視化のため、各光線の直線的な伝搬を z=0 から z = focal_length までプロット
    z_vals = np.linspace(0, focal_length, 100)

    for i in range(num_rays):
        x0 = rays_after[0, i]
        theta = rays_after[1, i]
        # 各光線の x(z) = x0 + z * theta
        x_line = x0 + z_vals * theta
        ax.plot(z_vals, x_line, label=f'input x={x_inputs[i]:.1f} mm')

    # 焦点面の位置における収束点を点でプロット
    ax.scatter([focal_length]*num_rays, x_focal, color='red', zorder=5, label='intersection at focal plane')
    ax.axvline(focal_length, color='red', linestyle='--', label=f'focal plane z = {focal_length:.1f} mm')

    ax.set_xlabel('z axis (mm)')
    ax.set_ylabel('Ray height x (mm)')
    ax.set_title('Ray tracing with thick lens approximation (simulation using ABCD matrix)')
    ax.grid(True)
    ax.set_xlim(min(z_front), z_lim)


# --- レンズシステムのパラメータ設定 ---
# ここではシンプルな厚レンズ（2面レンズ）を例とする
# サンプルパラメータ（単位は mm）
n_air = 1.0          # 空気の屈折率
n_glass = 1.5        # レンズ材（ガラス）の屈折率
R1 = 50.0            # 第1面の曲率半径（正：凸面, 物体側から見て凸）
R2 = -50.0           # 第2面の曲率半径（負：右側から見て凸；標準的な厚レンズの符号系）
d = 10.0             # レンズ内の厚さ

z_lim = +100

fig = plt.figure(figsize=(8, 5))
n_glass = [1.5, 1.7, 2.0]
count = len(n_glass)
row_plot = 6
for index, _n_glass in enumerate(n_glass):
    ax = fig.add_subplot(row_plot, count, index+1)
    camera_lens(ax, n_air, _n_glass, R1, R2, d, z_lim)
plt.show()
