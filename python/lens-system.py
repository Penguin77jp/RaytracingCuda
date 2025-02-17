import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
import streamlit as st

DIR_NUM = st.sidebar.slider("Number of Directions (DIR_NUM)", min_value=1, max_value=200, value=10)
LENS_NORMAL_LENGTH = st.sidebar.slider("Lens Normal Length", min_value=0, max_value=20, value=5)
OVER_RAY_LENGTH = st.sidebar.slider("Over Ray Length", min_value=0, max_value=100, value=5)

@dataclass
class Lens:
    """
    単一レンズを表現するクラス
    曲率（半径）、直径、位置、屈折率を持つ
    """
    curvature: float  # Radius of curvature
    diameter: float
    position: float
    refractive_index: float

    @property
    def focal_length(self) -> float:
        """
        プラノコンベックスレンズの場合の薄レンズ公式から焦点距離を計算:
           f = R/(n-1)
        """
        if abs(self.refractive_index - 1) < 1e-6:
            return float('inf')
        return self.curvature / (self.refractive_index - 1)

    def refract(self, ray_dir: np.ndarray, n1: float, n2: float, intersection: np.ndarray) -> np.ndarray:
        """
        スネルの法則に基づいて光線の屈折を計算
        n1: 入射側媒介の屈折率
        n2: 出射側媒質の屈折率
        intersection: レンズ面での交点
        """
        # 修正後: 球面中心をレンズ位置 + curvature として計算
        center = np.array([self.position + self.curvature, 0])
        # 交点から中心へのベクトル → これが面の法線（外向き）
        normal = intersection - center
        normal = normal / np.linalg.norm(normal)
        
        # もしレイと法線が同じ向きなら法線を反転
        if np.dot(ray_dir, normal) > 0:
            normal = -normal

        # デバッグ出力: 交点と法線情報
        print("Intersection:", intersection, "Normal:", normal)
        # 現在のAxesを取得し、矢印を描画する（プロット中の場合）
        ax = plt.gca()
        if ax is not None:
            # 交点から法線方向に長さ5の矢印を描画（色: magenta）
            if LENS_NORMAL_LENGTH > 0:
                ax.arrow(intersection[0], intersection[1], normal[0]*LENS_NORMAL_LENGTH, normal[1]*LENS_NORMAL_LENGTH,
                        head_width=2, head_length=3, fc='magenta', ec='magenta')
                ax.scatter(intersection[0], intersection[1], color='magenta', zorder=10)

        # 入射光線と法線の内積で cosθi を計算
        cos_theta_i = -np.dot(ray_dir, normal)
        sin_theta_i = np.sqrt(max(0, 1 - cos_theta_i**2))
        
        eta = n1 / n2
        print('eta:', eta, 'n1:', n1, 'n2:', n2)
        sin_theta_t = eta * sin_theta_i
        # 全反射の場合は反射光として返す
        if sin_theta_t > 1.0:
            return ray_dir  
        cos_theta_t = np.sqrt(max(0, 1 - sin_theta_t**2))
        
        # Tait’s form
        refracted = eta * ray_dir + (eta * cos_theta_i - cos_theta_t) * normal
        return refracted

    def apply_thin_lens_formula(self, ray_pos: np.ndarray, ray_dir: np.ndarray, current_refractive_index: float) -> np.ndarray:
        """
        薄レンズの公式を適用して光線の方向を変更
        入射光線の方向と高さを考慮して屈折後の方向を計算
        """
        h = ray_pos[1]  # レンズ面での高さ
        if abs(h) < 1e-10:  # 光軸上の場合
            return ray_dir
            
        # 入射角を考慮した角度計算
        new_dir = self.refract(ray_dir, current_refractive_index, self.refractive_index, ray_pos)
        
        return new_dir / np.linalg.norm(new_dir)

class LensSystem:
    """
    複数レンズから構成される光学システム
    光線追跡による像形成をシミュレート
    """
    
    def __init__(self, lenses: List[Lens], sensor_distance: float, debug: bool = False):
        self.lenses = sorted(lenses, key=lambda x: x.position)
        self.sensor_distance = sensor_distance
        self.debug = debug
        if self.sensor_distance is None:
            self.sensor_distance = self.focal_length() * 2

    def focal_length(self) -> float:
        """
        全体の焦点距離を計算
        """
        f = 0.0
        for lens in self.lenses:
            f += lens.focal_length
        return f
    def intersect_ray_circle(self, ray_origin: np.ndarray, ray_dir: np.ndarray, center: np.ndarray, radius: float) -> float:
        """
        円 (center, radius) とレイ (ray_origin, ray_dir) の交点パラメータ t を計算する。
        レイは r(t) = ray_origin + t * ray_dir  (t >= 0) と表す。
        
        交点が存在しない場合は None を返す。
        """
        # ray_direction は正規化されている前提
        A = 1.0
        B = 2 * np.dot(ray_origin - center, ray_dir)
        C = np.dot(ray_origin - center, ray_origin - center) - radius**2
        
        discriminant = B**2 - 4*A*C
        if discriminant < 0:
            return None

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-B - sqrt_disc) / (2*A)
        t2 = (-B + sqrt_disc) / (2*A)
        
        # 正の t のうち、小さい方を返す
        ts = [t for t in (t1, t2) if t > 1e-6]
        if not ts:
            return None
        return min(ts)
        
    def trace_ray(self, ray_origin: Tuple[float, float], ray_direction: Tuple[float, float]) -> List[Tuple[float, float]]:
        points = [ray_origin]
        current_pos = np.array(ray_origin)
        current_dir = np.array(ray_direction) / np.linalg.norm(ray_direction)
        
        no_valid_intersection_t = OVER_RAY_LENGTH
        current_refractive_index = 1.0
        for lens_index, lens in enumerate(self.lenses):
            print(f"--- Lens {lens_index}: current_pos = {current_pos}, current_dir = {current_dir}")
            R = abs(lens.curvature)
            # 修正: 球面中心をレンズ位置 + curvature として計算 (refractと統一)
            center = np.array([lens.position + lens.curvature, 0])
            t_val = self.intersect_ray_circle(current_pos, current_dir, center, R)
            if t_val is None:
                print(f"Lens {lens_index}: No intersection found. Using fallback ray length: {no_valid_intersection_t}")
                intersection = current_pos + no_valid_intersection_t * current_dir
                points.append(tuple(intersection))
                return points
            print(f"Lens {lens_index}: t_val = {t_val}")
            intersection = current_pos + t_val * current_dir
            print(f"Lens {lens_index}: Calculated intersection = {intersection} (half aperture = {lens.diameter/2})")
            
            if abs(intersection[1]) > lens.diameter / 2:
                print(f"Lens {lens_index}: Ray blocked by iris/aperture. intersection y = {intersection[1]} exceeds limit {lens.diameter/2}")
                intersection = current_pos + no_valid_intersection_t * current_dir
                points.append(tuple(intersection))
                return points
                
            points.append(tuple(intersection))
            # レンズによる屈折計算
            current_dir = lens.apply_thin_lens_formula(intersection, current_dir, current_refractive_index)
            current_pos = intersection
            current_refractive_index = lens.refractive_index

        final_t = (self.sensor_distance - current_pos[0]) / current_dir[0]
        final_point = current_pos + final_t * current_dir
        points.append(tuple(final_point))
        return points

    def calculate_image_point(self, object_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        物体点から像点を計算
        複数の光線の交点を平均して像点を求める
        """
        angles = np.linspace(-np.pi/6, np.pi/6, 10)
        image_points = []
        
        if self.debug:
            print("=== calculate_image_point start ===")
            print("Object point:", object_point)
        
        for angle in angles:
            direction = (np.cos(angle), np.sin(angle))
            ray_path = self.trace_ray(object_point, direction)
            if len(ray_path) > 1:
                image_points.append(ray_path[-1])
        
        if not image_points:
            return None
            
        image_point = tuple(np.mean(image_points, axis=0))
        if self.debug:
            print("Final image point:", image_point)
            print("=== calculate_image_point end ===\n")
        return image_point

def example_usage():
    # レンズパラメータを曲率で定義（プラノコンベックスの場合）
    # 例: 焦点距離 100 mm のレンズ → curvature = 100 * (n-1) = 50 (n=1.5)
    lenses = [
        Lens(curvature=50.0, diameter=40.0, position=100.0),
        # 焦点距離 150 mm のレンズ → curvature = 150 * (n-1) = 75
        Lens(curvature=75.0, diameter=50.0, position=200.0)
    ]
    sensor_distance = 350.0
    system = LensSystem(lenses, sensor_distance, debug=True)
    
    object_point = (0.0, 5.0)
    image_point = system.calculate_image_point(object_point)
    return image_point

def plot_lens_system(system: LensSystem, object_points: List[Tuple[float, float]]):
    """
    Visualize the lens system and the rays from multiple object points.
    Now the lens is plotted as an arc showing its curvature.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw lenses as curved arcs or vertical lines when curvature is 0 (iris/aperture)
    for lens in system.lenses:
        if abs(lens.curvature) < 1e-6:
            # 絞りの場合は縦線で表現 (lens.position)
            ys = np.linspace(-lens.diameter/2, lens.diameter/2, 100)
            xs = np.full_like(ys, lens.position)
        else:
            ys = np.linspace(-lens.diameter/2, lens.diameter/2, 100)
            if lens.curvature >= 0:
                # 修正後: Convex surface の場合
                xs = lens.position + lens.curvature - np.sqrt(np.clip(lens.curvature**2 - ys**2, 0, None))
            else:
                # 修正後: Concave surface の場合
                xs = lens.position + lens.curvature + np.sqrt(np.clip(lens.curvature**2 - ys**2, 0, None))
            
        ax.plot(xs, ys, color='blue', lw=2, label='Lens' if lens == system.lenses[0] else "")
        if abs(lens.curvature) >= 1e-6:
            focal_point = lens.position + lens.focal_length
            print('focal_point', focal_point)
            ax.plot([focal_point], [0], 'r+', markersize=10)
            ax.text(focal_point, -5, f'Focal {lens.focal_length:.1f}',
                    ha='center', va='top', color='red')
        else:
            # 絞りの場合、焦点は描画しない
            ax.text(lens.position, -5, 'Iris', ha='center', va='top', color='red')
    
    # Draw sensor
    ax.plot([system.sensor_distance, system.sensor_distance],
            [-60, 60], color='green', linestyle='--', lw=2, label='Sensor')
    
    # Plot rays
    colors = ["red", "green", "blue"]
    for idx, object_point in enumerate(object_points):
        ax.scatter(*object_point, color=colors[idx], s=100, zorder=5,
                   label=f'Object {idx+1}')
        angles = np.linspace(-np.pi/6, np.pi/6, DIR_NUM)
        for angle in angles:
            direction = (np.cos(angle), np.sin(angle))
            ray_path = system.trace_ray(object_point, direction)
            xs, ys = zip(*ray_path)
            ax.plot(xs, ys, color=colors[idx], alpha=0.7)
    
    ax.set_xlabel('Optical Axis (mm)')
    ax.set_ylabel('Height (mm)')
    ax.set_title('Optimized Lens System from Multiple Object Points')
    ax.set_xlim(object_points[0][0] - 20, system.sensor_distance + 20)
    ax.set_ylim(-60, 60)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

def exemple_tessar():
    lens = [
        Lens(curvature=50.0, diameter=30.0, position=0.0, refractive_index=1.5168),
        Lens(curvature=-50.0, diameter=25.0, position=15.0, refractive_index=1.6727),
        Lens(curvature=50.0, diameter=25.0, position=17.0, refractive_index=1.5168),
        Lens(curvature=50.0, diameter=20.0, position=35.0, refractive_index=1.5168),
        Lens(curvature=0.0, diameter=10.0, position=25.0, refractive_index=1.0),
    ]
    sensor_distance = 50
    system = LensSystem(lens, sensor_distance, debug=True)
    return system

def exeple_double_convex_lens(object_focal_length):
    """
{
  "lens_type": "Double Convex Lens",
  "material": "N-BK7",
  "dimensions": {
    "diameter": 25.4,
    "center_thickness": 5.0,
    "edge_thickness": 2.0
  },
  "curvature_radius": {
    "front_surface": 50.0,
    "back_surface": 50.0
  },
  "focal_length": 100.0,
  "refractive_index": 1.5168,
  "coating_options": [
    {
      "type": "Uncoated",
      "wavelength_range": "350 nm - 2.0 µm"
    },
    {
      "type": "AR Coating A",
      "wavelength_range": "350 - 700 nm"
    },
    {
      "type": "AR Coating B",
      "wavelength_range": "650 - 1050 nm"
    }
  ],
  "surface_quality": {
    "irregularity": "<λ/4 @632.8nm",
    "scratch_dig": "60/40 (standard), 10/5 (high precision)"
  },
  "applications": [
    "Beam focusing",
    "Imaging systems",
    "Optical instruments"
  ],
  "unit": "mm"
}

    """
    lens = [
        Lens(curvature=50.0, diameter=25.4, position=0.0, refractive_index=1.5168),
        Lens(curvature=-50.0, diameter=25.4, position=5.0, refractive_index=1.0),
        # Lens(curvature=0.0, diameter=5.0, position=25.0, refractive_index=1.0),
    ]
    focal_length = 100.0
    sensor_distance = object_focal_length * focal_length / (object_focal_length - focal_length + 10e-6)
    system = LensSystem(lens, sensor_distance, debug=True)
    return system

# Streamlit用メイン処理
st.title("Lens System Simulation")
print('\n\n\n\n\n')
print("=== Start ===")
object_points = [(-300.0, 10.0), (-200.0, 0.0), (-150.0, -10.0)]
system = exeple_double_convex_lens(object_focal_length=300)
plot_lens_system(system, object_points)
