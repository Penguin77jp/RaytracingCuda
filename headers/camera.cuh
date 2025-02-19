#pragma once

#include "camera.cuh"
#include "ray.cuh"
#include "util_json.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <filesystem>
#include <iostream>
#include <cstdio>

using json = nlohmann::json;

class Lens {
public:
	Lens(const float curvature, const float diameter, const float thickness, const float aperture, const float refractive_index) {
		this->curvature = curvature;
		this->diameter = diameter;
		this->thickness = thickness;
		this->aperture = aperture;
		this->refractive_index = refractive_index;

		if (abs(this->refractive_index - 1) < 1e-6) {
			this->focal_length = INFINITY;
		}
		else {
			this->focal_length = this->curvature / (this->refractive_index - 1);
		}
	}
	float curvature;
	float diameter;
	float thickness;
	float aperture;
	float refractive_index;
	float focal_length;
};

class LensSystem {
public:
	LensSystem(Lens* lenses, const int num_lenses) {
		this->lenses = lenses;
		this->num_lenses = num_lenses;


		// compute focal length
		// https://www.optics-words.com/kikakogaku/combined-focal-length.html
		{
			float tmp_f_sum = 0.0;
			for (int i = 0; i < num_lenses; ++i) {
				tmp_f_sum += 1.0 / lenses[i].focal_length;
			}
			float tmp_f_dash_sum = 0.0;
			for (int i = 0; i < num_lenses - 1; ++i) {
				tmp_f_dash_sum += this->lenses[i].thickness / this->lenses[i].focal_length / this->lenses[i + 1].refractive_index;
			}
			//this->focal_length = tmp_f_sum - tmp_f_dash_sum;
		}

		// compute image distance
		this->image_distance = -1.0f;

	}
	LensSystem(const std::string& json_file) {

	}

	__device__
		Ray get_ray(const float screen_u, const float screen_v, curandState* rand_state) const {
		Ray ray;
		for (int lens_index = 0; lens_index < num_lenses; ++lens_index) {

		}
		return Ray();
	}
	float compute_focal_length() const {
		exit(1);
		return 0.0f;
	}

	Lens* lenses;
	int num_lenses;
	float image_distance;
};

class Camera {
public:
	Camera(const Vec3 position, const Vec3 direction, const Vec3 up, const int width, const int height, LensSystem* lens_system) {
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
		if (this->lens_system == nullptr) {
			// レイの方向計算
			const Vec3& cam_direction = -direction_z;
			Vec3 screen_point = cam_direction * focal_length + direction_x * screen_u + direction_y * scnree_v;

			ray.origin = this->position;
			ray.direction = normalize(screen_point);
		}
		else {
			const Ray lens_system_ray = this->lens_system->get_ray(screen_u, scnree_v, &local_rand_state);
			ray = lens_system_ray;
		}
		return ray;
	}

	Vec3 position;
	Vec3 direction;
	int width, height;
	float focal_length;
	Vec3 up;
	LensSystem* lens_system = nullptr;
};

