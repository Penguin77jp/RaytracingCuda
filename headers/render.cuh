#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "ray.cuh"
#include <highfive/H5File.hpp>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include "camera.cuh"

#define RAY_SAMPLES 1e5
#define PI 3.14159265358979323846f
#define RAY_EPSILON 1.0e-6f
#define MAX_TRACE_DEPTH 24
#define MAX_TRIANGLES 10000

struct Sphere {
	Vec3 center;
	float radius;
	Vec3 color;
	Vec3 emission;
};

struct Triangle {
	Vec3 v0, v1, v2;
	Vec3 normal;
	Vec3 color;
	Vec3 emission;
};

__device__
float hit_sphere(const Vec3& center, const float radius, const Ray& ray) {
	Vec3 oc = { ray.origin.x - center.x,
			  ray.origin.y - center.y,
			  ray.origin.z - center.z };
	const float a = length_squared(ray.direction);
	const float b = 2.0f * dot(ray.direction, oc);
	const float c = length_squared(oc) - radius * radius;

	const float discriminant = b * b - 4 * a * c;
	if (discriminant < 0) {
		return -1.0f;
	}
	const float sqrt_d = sqrtf(discriminant);
	const float t1 = (-b + sqrt_d) / (2 * a);
	const float t2 = (-b - sqrt_d) / (2 * a);
	// 物理的に意味のある正の解を選ぶ
	if (t1 > 0 && t2 < 0)
		return t1;
	else if (t1 < 0 && t2 > 0)
		return t2;
	else
		if (t1 >= 0 && t2 >= 0)
			return fminf(t1, t2);


	return -1.0;
}

__device__
float hit_triangle(const Triangle& triangle, const Ray& ray, Vec3& out_normal) {
	Vec3 edge1 = triangle.v1 - triangle.v0;
	Vec3 edge2 = triangle.v2 - triangle.v0;
	Vec3 h = cross(ray.direction, edge2);
	float a = dot(edge1, h);
	if (fabs(a) < 1e-6f)
		return -1.0f; // レイは平行

	float f = 1.0f / a;
	Vec3 s = ray.origin - triangle.v0;
	float u = f * dot(s, h);
	if (u < 0.0f || u > 1.0f)
		return -1.0f;

	Vec3 q = cross(s, edge1);
	float v = f * dot(ray.direction, q);
	if (v < 0.0f || u + v > 1.0f)
		return -1.0f;

	float t = f * dot(edge2, q);
	if (t > RAY_EPSILON) {
		out_normal = triangle.normal;
		return t;
	}
	else
		return -1.0f;
}

struct Scene {
#define MAX_SPHERES 10
	Sphere spheres[MAX_SPHERES];
	int num_spheres;
	Triangle triangles[MAX_TRIANGLES];
	int num_triangles;
};

__device__
Vec3 reflect(const Vec3& v, const Vec3& n) {
	return v - n * 2.0f * dot(v, n);
}

__device__
Vec3 diffuse(const Vec3& normal, curandState& rand_state) {
	//curand_uniform(&local_rand_state)
	const float r1 = 2.0f * PI * (float)curand_uniform(&rand_state);
	const float r2 = curand_uniform(&rand_state);
	const float r2s = sqrtf(r2);
	Vec3 w = normal;
	Vec3 u = normalize(cross((fabs(w.x) > 0.1 ? Vec3{ 0.0f, 1.0f, 0.0f } : Vec3{ 1.0f, 0.0f, 0.0f }), w));
	Vec3 v = cross(w, u);
	return normalize(u * cosf(r1) * r2s + v * sinf(r1) * r2s + w * sqrtf(1 - r2));
}

__device__
float hit_scene(
	const Scene& scene,
	const Ray& ray,
	const int prev_hit_sphere_id,
	const int prev_hit_triangle_id,
	Vec3& hit_point,
	Vec3& normal,
	int& hit_sphere_id,
	int& hit_triangle_id
) {
	float t_min = FLT_MAX;
	int which_hit_object_type = -1; // 0: sphere, 1: triangle

	// intersect with spheres
	for (int i = 0; i < scene.num_spheres; ++i) {
		if (i == prev_hit_sphere_id) {
			continue;
		}
		float t = hit_sphere(scene.spheres[i].center, scene.spheres[i].radius, ray);
		if (t > RAY_EPSILON && t < t_min) {
			t_min = t;
			which_hit_object_type = 0;
			hit_point = ray.origin + ray.direction * t;
			normal = normalize(hit_point - scene.spheres[i].center);
			hit_sphere_id = i;
		}
	}

	// intersect with triangles
	for (int i = 0; i < scene.num_triangles; ++i) {
		if (i == prev_hit_triangle_id) {
			continue;
		}
		Vec3 tri_normal;
		float t = hit_triangle(scene.triangles[i], ray, tri_normal);
		if (t > RAY_EPSILON && t < t_min) {
			t_min = t;
			which_hit_object_type = 1;
			hit_point = ray.origin + ray.direction * t;
			normal = tri_normal;
			hit_triangle_id = i;
		}
	}

	if (which_hit_object_type == 0) {
		hit_triangle_id = -1;
	}
	else if (which_hit_object_type == 1) {
		hit_sphere_id = -1;
	}
	else {
		hit_sphere_id = -1;
		hit_triangle_id = -1;
	}

	if (which_hit_object_type != -1) {
		return t_min;
	}
	else {
		return -1.0f;
	}
}

__device__
Vec3 pathtracing_steradian_sample(Scene& scene, Ray ray, curandState& rand_state) {
	//if (depth >= MAX_TRACE_DEPTH) {
	//	// when trace loop is over the max depth
	//	return { 0.0f, 0.0f, 0.0f };
	//}

	//Vec3 hit_point, normal, color;
	//float t = hit_scene(scene, ray, hit_point, normal, color);
	//printf("depth:%d, t:%f\n", depth, t);

	//if (t > 0.0f) {
	//	Vec3 reflected_dir = reflect(ray.direction, normal);
	//	//return reflected_dir;
	//	Ray reflected_ray = { hit_point, reflected_dir };
	//	Vec3 reflected_color = trace_ray(scene, reflected_ray, depth + 1);
	//	return reflected_color;
	//	//return color * 0.5f + reflected_color * 0.5f;
	//}
	//else {
	//	// 背景色
	//	return { 0.5f, 0.7f, 1.0f };
	//}

	Vec3 hit_point, normal, final_color, attenuation_color;
	final_color = { 0.0f, 0.0f, 0.0f };
	attenuation_color = { 1.0f, 1.0f, 1.0f };
	int hit_sphere_id = -1, hit_triangle_id = -1;
	for (int depth = 0; depth < MAX_TRACE_DEPTH; ++depth) {
		float t = hit_scene(scene, ray, hit_sphere_id, hit_triangle_id, hit_point, normal, hit_sphere_id, hit_triangle_id);
		//return { (float)hit_sphere_id, (float)hit_triangle_id, t };
		if (t > 0.0f) {
			//Vec3 reflected_dir = reflect(ray.direction, normal);
			Vec3 reflected_dir = diffuse(normal, rand_state);
			ray.origin = hit_point;
			ray.direction = reflected_dir;
			Vec3 object_color = { 0.0f, 0.0f, 0.0f };
			Vec3 object_emission = { 0.0f, 0.0f, 0.0f };
			if (hit_sphere_id != -1) {
				object_color = scene.spheres[hit_sphere_id].color;
				object_emission = scene.spheres[hit_sphere_id].emission;
			}
			else if (hit_triangle_id != -1) {
				object_color = scene.triangles[hit_triangle_id].color;
				object_emission = scene.triangles[hit_triangle_id].emission;
			}
			else {
				// error
				return { -1.0f, -1.0f, -1.0f };
			}
			//return object_color;
			final_color = final_color + attenuation_color * object_emission;
			attenuation_color = attenuation_color * object_color;
		}
		else {
			//const Vec3 background_color = { 0.5f, 0.7f, 1.0f };
			break;
		}
	}
	return final_color;
}

__device__
Vec3 pathtracing_surface_sample(Scene& scene, Ray ray, curandState& rand_state) {
	Vec3 hit_point, normal_intersected, final_color, attenuation_color;
	final_color = { 0.0f, 0.0f, 0.0f };
	attenuation_color = { 1.0f, 1.0f, 1.0f };
	int sphere_id_intersected = -1, triangle_id_intersected = -1;
	float ray_t_intersected = hit_scene(scene, ray, sphere_id_intersected, triangle_id_intersected, hit_point, normal_intersected, sphere_id_intersected, triangle_id_intersected);
	if (ray_t_intersected < 0.0f) {
		return { 0.0f, 0.0f, 0.0f };
	}
	Vec3 current_point = ray.origin + ray.direction * ray_t_intersected;
	bool debug_print = false;
	if (sphere_id_intersected == 2) {
		//printf("sphere_id_intersected:%d, triangle_id_intersected:%d, ray_t_intersected:%f\n", sphere_id_intersected, triangle_id_intersected, ray_t_intersected);
		//debug_print = true;
	}
	float tmp_dot = dot(normal_intersected, -ray.direction);
	if (tmp_dot < 0.0f) {
		return { -1.f, -1.f, -1.f };
	}
	attenuation_color = attenuation_color * tmp_dot;


	//return normal_intersected;
	//for (int depth = 0; depth < 1; ++depth) {
	for (int depth = 0; depth < MAX_TRACE_DEPTH; ++depth) {
		Sphere& intersected_sphere = scene.spheres[sphere_id_intersected];

		const int& max_spheres = scene.num_spheres;
		const int random_sphere_id = (int)(curand_uniform(&rand_state) * max_spheres);
		const Sphere& random_sphere = scene.spheres[random_sphere_id];
		const float sphere_area = 4.0f * PI * random_sphere.radius * random_sphere.radius;
		Vec3 sample_point_in_surface;
		{
			float u = curand_uniform(&rand_state);
			float v = curand_uniform(&rand_state);
			float theta = 2.0f * PI * u;
			float phi = acosf(1.0f - 2.0f * v);
			constexpr float eps = 0.0;
			float x = (random_sphere.radius + eps) * sinf(phi) * cosf(theta);
			float y = (random_sphere.radius + eps) * sinf(phi) * sinf(theta);
			float z = (random_sphere.radius + eps) * cosf(phi);
			sample_point_in_surface = { random_sphere.center.x + x,
										random_sphere.center.y + y,
										random_sphere.center.z + z };
		}
		//return { dot(normal_intersected, normalize(sample_point_in_surface - current_point)), 0.0f, 0.0f };
		bool is_visible = false;
		Ray shadow_ray;
		Vec3 shadow_intersect_normal;
		float shadow_ray_t;
		int shadow_hit_sphere_id = -1, shadow_hit_triangle_id = -1;
		{
			Vec3 ray_dir = normalize(sample_point_in_surface - current_point);
			//return { dot(normal_intersected, ray_dir), 0.0f, 0.0f };
			shadow_ray = { current_point, ray_dir };
			//shadow_ray_t = hit_scene(scene, shadow_ray, -1, -1, hit_point, shadow_intersect_normal, shadow_hit_sphere_id, shadow_hit_triangle_id);
			shadow_ray_t = hit_scene(scene, shadow_ray, sphere_id_intersected, triangle_id_intersected, hit_point, shadow_intersect_normal, shadow_hit_sphere_id, shadow_hit_triangle_id);
			//return { shadow_ray_t, (float)shadow_hit_sphere_id, (float)shadow_hit_triangle_id };
			if (shadow_ray_t > 0.0f && shadow_hit_sphere_id == random_sphere_id) {
				is_visible = true;
			}
			else {
				is_visible = false;
			}
		}
		if (!is_visible) {
			final_color = final_color + attenuation_color * intersected_sphere.emission;
			break;
		}
		//return { dot(normal_intersected, shadow_ray.direction), 0.0f, 0.0f };
		const float cos_shadow_ray_current_point = dot(normal_intersected, shadow_ray.direction);
		const float cos_shadow_ray_sample_point = dot(-shadow_ray.direction, shadow_intersect_normal);
		if (cos_shadow_ray_current_point < 0.0f || cos_shadow_ray_sample_point < 0.0f) {
			final_color = final_color + attenuation_color * intersected_sphere.emission;
			break;
		}
		//return { cos_shadow_ray_current_point, cos_shadow_ray_sample_point, 0.0f };
		const float distance_to_sample_pow2 = length_squared(sample_point_in_surface - current_point);
		const float geometric_factor = 1.0f / distance_to_sample_pow2 * cos_shadow_ray_current_point * cos_shadow_ray_sample_point;
		const float diffuse_reflectance = 1.0f / PI;
		const float pdf = 1.0 / sphere_area;
		/*if (true && sphere_id_intersected == 2 && random_sphere_id == 0 && is_visible) {
			debug_print = true;
		}*/

		//if (debug_print) {
		//	// debug print in one line
		//	// cos_shadow_ray_current_point, cos_shadow_ray_sample_point, distance_to_sample_pow2, geometric_factor, pdf, is_visible, shadow_ray_t, shadow_hit_sphere_id, shadow_hit_triangle_id
		//	printf("depth:%d, sphere_id_intersected:%d, triangle_id_intersected:%d, random_sphere_id:%d, shadow_ray_t:%f, shadow_hit_sphere_id:%d, shadow_hit_triangle_id:%d, cos_shadow_ray_current_point:%f, cos_shadow_ray_sample_point:%f, distance_to_sample_pow2:%f, geometric_factor:%f, pdf:%f, is_visible:%d\n",
		//		depth,
		//		sphere_id_intersected,
		//		triangle_id_intersected,
		//		random_sphere_id,
		//		shadow_ray_t,
		//		shadow_hit_sphere_id,
		//		shadow_hit_triangle_id,
		//		cos_shadow_ray_current_point,
		//		cos_shadow_ray_sample_point,
		//		distance_to_sample_pow2,
		//		geometric_factor,
		//		pdf,
		//		is_visible);
		//}

		final_color = final_color + attenuation_color * (intersected_sphere.color * geometric_factor * diffuse_reflectance / pdf + intersected_sphere.emission);
		attenuation_color = attenuation_color * intersected_sphere.color;

		// set next ray
		current_point = shadow_ray.origin + shadow_ray.direction * (shadow_ray_t);
		sphere_id_intersected = random_sphere_id;
		triangle_id_intersected = -1;
	}
	return final_color;
}

__device__
void compute_photonic_ray(Scene& in_scene, PhotonicRay& out_ray) {

}

__device__
Vec3 pathtracing_bdpf(Scene& scene, Ray ray, curandState& rand_state) {
	constexpr int pthotic_ray_num = 10;
	PhotonicRay eye_payh[pthotic_ray_num];
	PhotonicRay light_path[pthotic_ray_num];
}

__global__
void render_init(curandState* rand_state, const int width, const int height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int pixel_index = y * width + x;
	if (x >= width || y >= height) {
		return;
	}
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__
void render_pixel(float* output, curandState* rand_state, Scene* scene, Camera cam, int render_mode) {
	const int width = cam.width;
	const int height = cam.height;
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int pixel_index = y * width + x;
	//printf("threadIdx:(%d, %d), blockIdx:(%d, %d),x:%d y:%d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, x, y);

	if (x >= width || y >= height) {
		//printf("thread out of range, x:%d, y:%d\n", x, y);
		return;
	}
	//printf("thread in range, x:%d, y:%d\n", x, y);

	Ray ray = cam.get_ray(x, y, &rand_state[pixel_index]);
	curandState local_rand_state = rand_state[pixel_index];

	//printf("ray.direction(%f, %f, %f)\n", ray.direction.x, ray.direction.y, ray.direction.z);

	Vec3 accumulated_color = { 0.0f, 0.0f, 0.0f };
	for (int i = 0; i < RAY_SAMPLES; ++i) {
		Vec3 color;
		if (render_mode == 0) {
			color = pathtracing_steradian_sample(*scene, ray, local_rand_state);
		}
		else if (render_mode == 1) {
			color = pathtracing_surface_sample(*scene, ray, local_rand_state);
		}
		else if (render_mode == 2) {
			constexpr int pthotic_ray_num = 10;
			PhotonicRay eye_payh[pthotic_ray_num];
			PhotonicRay light_path[pthotic_ray_num];

			color = pathtracing_bdpf(*scene, ray, local_rand_state);
		}
		else {
			//Vec3 color = { 0.0f, 0.0f, 0.0f };
		}
		accumulated_color = accumulated_color + color / (float)RAY_SAMPLES;
	}
	//accumulated_color = ray.direction;

	output[pixel_index * 3 + 0] = accumulated_color.x;
	output[pixel_index * 3 + 1] = accumulated_color.y;
	output[pixel_index * 3 + 2] = accumulated_color.z;


	//output[pixel_index * 3 + 0] = (float)t;

	//output[pixel_index * 3 + 0] = (float)ray.direction.x;
	//output[pixel_index * 3 + 1] = (float)ray.direction.y;
	//output[pixel_index * 3 + 2] = (float)ray.direction.z;

	//printf("pixel_index:%d, %d, x:%d, y:%d, value(%f, %f, %f)\n", pixel_index, pixel_index * 3, x, y, output[pixel_index * 3 + 0], output[pixel_index * 3 + 1], output[pixel_index * 3 + 2]);
}

__global__
void render_pixel_bdpf(float* output, curandState* rand_state, Scene* scene, Camera cam, int render_mode) {
	const int width = cam.width;
	const int height = cam.height;
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int pixel_index = y * width + x;
	//printf("threadIdx:(%d, %d), blockIdx:(%d, %d),x:%d y:%d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, x, y);

	if (x >= width || y >= height) {
		//printf("thread out of range, x:%d, y:%d\n", x, y);
		return;
	}
	//printf("thread in range, x:%d, y:%d\n", x, y);

	Ray ray = cam.get_ray(x, y, &rand_state[pixel_index]);
	curandState local_rand_state = rand_state[pixel_index];

	//printf("ray.direction(%f, %f, %f)\n", ray.direction.x, ray.direction.y, ray.direction.z);

	Vec3 accumulated_color = { 0.0f, 0.0f, 0.0f };
	for (int i = 0; i < RAY_SAMPLES; ++i) {
		Vec3 color;
		if (render_mode == 0) {
			color = pathtracing_steradian_sample(*scene, ray, local_rand_state);
		}
		else if (render_mode == 1) {
			color = pathtracing_surface_sample(*scene, ray, local_rand_state);
		}
		else if (render_mode == 2) {
			constexpr int pthotic_ray_num = 10;
			PhotonicRay eye_payh[pthotic_ray_num];
			PhotonicRay light_path[pthotic_ray_num];

			color = pathtracing_bdpf(*scene, ray, local_rand_state);
		}
		else {
			//Vec3 color = { 0.0f, 0.0f, 0.0f };
		}
		accumulated_color = accumulated_color + color / (float)RAY_SAMPLES;
	}
	//accumulated_color = ray.direction;

	output[pixel_index * 3 + 0] = accumulated_color.x;
	output[pixel_index * 3 + 1] = accumulated_color.y;
	output[pixel_index * 3 + 2] = accumulated_color.z;


	//output[pixel_index * 3 + 0] = (float)t;

	//output[pixel_index * 3 + 0] = (float)ray.direction.x;
	//output[pixel_index * 3 + 1] = (float)ray.direction.y;
	//output[pixel_index * 3 + 2] = (float)ray.direction.z;

	//printf("pixel_index:%d, %d, x:%d, y:%d, value(%f, %f, %f)\n", pixel_index, pixel_index * 3, x, y, output[pixel_index * 3 + 0], output[pixel_index * 3 + 1], output[pixel_index * 3 + 2]);
}

using Image = std::vector<std::vector<std::vector<double>>>;

//template <typename T>
Image FlattenArray2Image(std::vector<double>& array, const int width, const int height, const int channel) {
	Image image(width, std::vector<std::vector<double>>(height, std::vector<double>(channel, 0)));
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int k = 0; k < channel; k++) {
				image[x][y][k] = array[y * width * channel + x * channel + k];
				//printf("x:%d, y:%d, k:%d, value(%f)\n", x, y, k, image[x][y][k]);
			}
			// print rgb value
			//printf("x:%d, y:%d, value(%f, %f, %f)\n", x, y, image[x][y][0], image[x][y][1], image[x][y][2]);
		}
	}
	return image;
}

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

