#pragma once

#include "ray.cuh"
#include "util_json.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <filesystem>
#include <iostream>
#include <cstdio>

using json = nlohmann::json;

#define LENS_SYSTEM_MAX_LENSES 10

// defined in render.cuh
__device__
float hit_sphere(const Vec3& center, const float radius, const Ray& ray);
// end of defined in render.cuh


// 屈折計算ヘルパー関数
inline __device__
bool refract(const Vec3& wi, const Vec3& n, float eta, Vec3& wt) {
	float cos_theta_i = dot(wi, n);
	float sin2_theta_i = fmaxf(1.0f - cos_theta_i * cos_theta_i, 0.0f);
	float sin2_theta_t = eta * eta * sin2_theta_i;

	if (sin2_theta_t >= 1.0f) return false; // 全反射

	float cos_theta_t = sqrtf(1.0f - sin2_theta_t);
	wt = eta * wi + (eta * cos_theta_i - cos_theta_t) * n;
	return true;
}

class Lens {
public:
	Lens(const float radius, const float diameter, const float thickness, const float refractive_index) {
		this->radius = radius;
		this->diameter = diameter;
		this->thickness = thickness;
		this->refractive_index = refractive_index;
	}
	Lens() {};

	__device__
		void propagate_ray(const Ray& in_ray, float& out_t, Ray& out_ray) const {
		const Vec3 center = { 0.0f, 0.0f, -thickness / 2 }; // レンズ中心位置

		const float t = hit_sphere(center, radius, in_ray);

		// 交点座標
		Vec3 point = in_ray.origin + in_ray.direction * out_t;

		// レンズ直径チェック
		if (sqrtf(point.x * point.x + point.y * point.y) > diameter / 2) {
			out_t = -1.0f;
			return;
		}

		// 法線ベクトル計算
		Vec3 normal = normalize(point - center);
		if (radius < 0) normal = -normal; // 凹面の場合

		// 屈折率の比 (現在の媒質からレンズ材質へ)
		float eta = refractive_index;

		// 屈折方向計算
		Vec3 direction;
		if (!refract(in_ray.direction, normal, eta, direction)) {
			// 全反射の場合
			out_t = -1.0f;
			return;
		}

		out_ray.origin = point;
		out_ray.direction = normalize(direction);
	}

	float radius;
	float diameter;
	float thickness;
	float refractive_index;
};

class Camera;
class LensSystem {
public:
	LensSystem(const std::string& json_file, const Camera& cam);
	LensSystem():num_lenses(0), object_focal_length(0.0f) {}
	void print() const;
	__device__
	bool valid() const {
		return num_lenses > 0;
	}

	__device__
		bool get_ray(const float screen_u, const float screen_v, curandState* rand_state, Ray& out_ray) const {
		const Vec3 image_size = { 1.0f, 1.0f, 0.0f };

		Ray ray_in_camera;
		ray_in_camera.origin = Vec3{ image_size.x * screen_u, image_size.y * screen_v, this->distance_to_image_plane };
		// random
		//ray_in_camera.direction = Vec3{ curand_uniform(rand_state) * 2.0f - 1.0f, curand_uniform(rand_state) * 2.0f - 1.0f, -1.0f };

		// DEBUG. The ray direction is always parallel to the z-axis.
		ray_in_camera.direction = Vec3{ 0.0f, 0.0f, -1.0f };
		Ray tmp_out_ray;
		for (int lens_index = 0; lens_index < num_lenses; ++lens_index) {
			const Lens& lens = lenses[lens_index];
			float t;
			lens.propagate_ray(ray_in_camera, t, tmp_out_ray);
			if (t < 0) {
				return false;
			}
			ray_in_camera = tmp_out_ray;
		}
		out_ray = tmp_out_ray;
		return true;
	}

	inline LensSystem& operator= (const LensSystem& lens_system) {
		for (int i = 0; i < lens_system.num_lenses; ++i) {
			this->lenses[i] = lens_system.lenses[i];
		}
		this->num_lenses = lens_system.num_lenses;
		this->image_distance = lens_system.image_distance;
		this->focal_length = lens_system.focal_length;
		this->distance_to_image_plane = lens_system.distance_to_image_plane;
		this->front_principal_plane = lens_system.front_principal_plane;
		this->back_principal_plane = lens_system.back_principal_plane;

		return *this;
	}

	Lens lenses[LENS_SYSTEM_MAX_LENSES];
	int num_lenses;
	float image_distance;
	float focal_length;
	const float& object_focal_length;
	float distance_to_image_plane;
	float front_principal_plane;
	float back_principal_plane;
};

class Camera {
public:
	Camera(const Vec3 position, const Vec3 direction, const Vec3 up, const int width, const int height, LensSystem lens_system) {
		this->position = position;
		this->direction = direction;
		this->up = up;
		this->lens_system = lens_system;
		this->focal_length = -1.0f;
		this->width = width;
		this->height = height;
	}
	Camera(const std::string& json_file);

	__device__
		void get_direction_xyz(Vec3& x, Vec3& y, Vec3& z) const {
		z = -this->direction;
		x = normalize(cross(this->up, z));
		y = cross(z, x);
	}
	void print_camera() const {
		printf("Camera position(%f, %f, %f)\n", this->position.x, this->position.y, this->position.z);
		//printf("Camera direction(%f, %f, %f)\n", cam.direction.x, cam.direction.y, cam.direction.z);
		printf("Camera direction(%f, %f, %f)\n", this->direction.x, this->direction.y, this->direction.z);
		printf("Camera up(%f, %f, %f)\n", this->up.x, this->up.y, this->up.z);
		printf("Camera lens_system(%p)\n", this->lens_system);
	}

	void look_at(const Vec3& target) {
		Vec3 direction = normalize({ target.x - this->position.x,
			target.y - this->position.y,
			target.z - this->position.z });
		this->direction = direction;
	}

	__device__
		Ray get_ray(const int x, const int y, curandState* rand_state) const {
		const int pixel_index = y * this->width + x;
		// カメラ設定
		const float aspect_ratio = this->width / (float)this->height;
		//printf("aspect_ratio:%f\n", aspect_ratio);
		const float focal_length = this->focal_length;
		//printf("focal_length:%f\n", focal_length);

		Vec3 direction_x, direction_y, direction_z;
		get_direction_xyz(direction_x, direction_y, direction_z);

		curandState local_rand_state = rand_state[pixel_index];

		// スクリーン上の位置（u,v座標系）
		const float screen_u = x / (float)this->width * 2.0f - 1.0f;
		const float scnree_v = (y / (float)this->height * 2.0f - 1.0f) / aspect_ratio;

		Ray ray;
		if (!lens_system.valid()) {
			// レイの方向計算
			const Vec3& cam_direction = -direction_z;
			Vec3 screen_point = cam_direction * focal_length + direction_x * screen_u + direction_y * scnree_v;

			ray.origin = this->position;
			ray.direction = normalize(screen_point);
		}
		else {
			Ray lens_system_ray;
			bool is_valid = lens_system.get_ray(screen_u, scnree_v, &local_rand_state, lens_system_ray);
			if (!is_valid) {
				ray.origin = { 0.0f, 0.0f, 0.0f };
				ray.direction = { 0.0f, 0.0f, 0.0f };
			}
			else {
				ray = lens_system_ray;
				//printf("ray.origin(%f, %f, %f)\n", ray.origin.x, ray.origin.y, ray.origin.z);
				//printf("ray.direction(%f, %f, %f)\n", ray.direction.x, ray.direction.y, ray.direction.z);
			}
		}
		return ray;
	}

	Vec3 position;
	Vec3 direction;
	int width, height;
	float focal_length;
	Vec3 up;
	LensSystem lens_system;
};

