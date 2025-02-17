#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "camera.cuh"
#include "ray.cuh"

#include <highfive/H5File.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <chrono>

//#define WIDTH 4
//#define HEIGHT 2
#define WIDTH 800
#define HEIGHT 600
#define RAY_SAMPLES 1e3
#define PI 3.14159265358979323846f
#define RAY_EPSILON 1.0e-6f
#define MAX_TRACE_DEPTH 10
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
float hit_sphere(Sphere sphere, Ray ray) {
	Vec3 oc = { ray.origin.x - sphere.center.x,
			  ray.origin.y - sphere.center.y,
			  ray.origin.z - sphere.center.z };
	const float a = length_squared(ray.direction);
	const float b = 2.0f * dot(ray.direction, oc);
	const float c = length_squared(oc) - sphere.radius * sphere.radius;

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
		float t = hit_sphere(scene.spheres[i], ray);
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
Vec3 trace_ray(Scene& scene, Ray ray, curandState& rand_state) {
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
		//return { (float)hit_object_id, (float)hit_triangle_id, t };
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

__global__
void render_init(curandState* rand_state) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int pixel_index = y * WIDTH + x;
	if (x >= WIDTH || y >= HEIGHT) {
		return;
	}
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__
void render_pixel(float* output, curandState* rand_state, Scene* scene, Camera cam) {
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int pixel_index = y * WIDTH + x;
	//printf("threadIdx:(%d, %d), blockIdx:(%d, %d),x:%d y:%d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, x, y);

	if (x >= WIDTH || y >= HEIGHT) {
		//printf("thread out of range, x:%d, y:%d\n", x, y);
		return;
	}
	//printf("thread in range, x:%d, y:%d\n", x, y);

	Ray ray = cam.get_ray(x, y, &rand_state[pixel_index]);
	curandState local_rand_state = rand_state[pixel_index];

	//printf("ray.direction(%f, %f, %f)\n", ray.direction.x, ray.direction.y, ray.direction.z);

	Vec3 accumulated_color = { 0.0f, 0.0f, 0.0f };
	for (int i = 0; i < RAY_SAMPLES; ++i) {
		Vec3 color = trace_ray(*scene, ray, local_rand_state);
		accumulated_color = accumulated_color + color / (float)RAY_SAMPLES;
	}

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

//// スカラ型（double, intなど）用のオーバーロード
//template <typename T>
//std::vector<rsize_t> getSize(const T&) {
//	return { 1 };
//}
//
//// 再帰的にサイズを取得するテンプレート特殊化
//
//template <typename T>
//std::vector<size_t> getSize(const std::vector<T>& container) {
//	if (container.empty()) return {};
//	//return container.size() * getSize(container[0]);
//	std::cout << "container.size():" << container.size() << std::endl;
//	const std::vector<size_t> inner = getSize(container[0]);
//	//return { container.size() }.insert(inner.begin(), inner.end());
//}

int main() {
	// check cuda init
	checkCudaErrors(cudaDeviceReset());

	float* d_output;
	checkCudaErrors(cudaMallocManaged(&d_output, WIDTH * HEIGHT * 3 * sizeof(float)));

	//Sphere sphere = { {0.0f, 0.0f, 0.0f}, 1.0f, {0.6f, 0.4f, 0.2f} };
	Scene h_scene;
	h_scene.num_spheres = 0;
	h_scene.num_triangles = 0;

	if (false) {
		h_scene.num_spheres = 4;
		h_scene.spheres[0] = { {0.0f, 0.0f, 0.0f}, 1.0f, {0.6f, 0.4f, 0.2f}, {0.0f, 0.0f, 0.0f} };
		h_scene.spheres[1] = { {2.0f, 0.0f, -1.0f}, 1.0f, {0.2f, 0.4f, 0.6f}, {0.0f, 0.0f, 0.0f} };
		// ground
		h_scene.spheres[2] = { {0.0f, -1000.0f - 10.0f, 0.0f}, 1000.0f, {0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 0.0f} };
		// light
		h_scene.spheres[3] = { {0.0f, 1.5f, 0.0f}, 0.25f, {0.0f, 0.0f, 0.0f}, {10.0f, 10.0f, 10.0f} };
	}
	else if (true) {
		// light and ground
		h_scene.num_spheres = 2;
		//h_scene.spheres[0] = { {0.0f, 0.0f, 0.0f}, 8.0f, {0.0f, 0.0f, 0.0f}, {10.0f, 10.0f, 10.0f} };
		h_scene.spheres[0] = { {0.0f, 3.5f + 5.0f, 2.0f}, 5.0f, {0.0f, 0.0f, 0.0f}, {10.0f, 10.0f, 10.0f} };
		h_scene.spheres[1] = { {0.0f, -1000.0f - 4.0f, 0.0f}, 1000.0f, {0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 0.0f} };
	}

	if (true) {
		std::string inputfile = "model.obj";
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn;
		std::string err;

		bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str());
		std::cout << "obj warning:" << warn << std::endl;
		if (!err.empty()) {
			std::cerr << err << std::endl;
		}
		if (!ret) {
			exit(1);
		}

		// 三角形のデータをシーンに格納
		Vec3 obj_bb_min = { FLT_MAX, FLT_MAX, FLT_MAX };
		Vec3 obj_bb_max = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
		for (size_t s = 0; s < shapes.size(); ++s) {
			size_t index_offset = 0;
			for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f) {
				int fv = shapes[s].mesh.num_face_vertices[f];
				if (fv != 3) continue; // 三角形のみを対象

				tinyobj::index_t idx0 = shapes[s].mesh.indices[index_offset + 0];
				tinyobj::index_t idx1 = shapes[s].mesh.indices[index_offset + 1];
				tinyobj::index_t idx2 = shapes[s].mesh.indices[index_offset + 2];

				Vec3 v0 = { attrib.vertices[3 * idx0.vertex_index + 0],
							attrib.vertices[3 * idx0.vertex_index + 1],
							attrib.vertices[3 * idx0.vertex_index + 2] };
				Vec3 v1 = { attrib.vertices[3 * idx1.vertex_index + 0],
							attrib.vertices[3 * idx1.vertex_index + 1],
							attrib.vertices[3 * idx1.vertex_index + 2] };
				Vec3 v2 = { attrib.vertices[3 * idx2.vertex_index + 0],
							attrib.vertices[3 * idx2.vertex_index + 1],
							attrib.vertices[3 * idx2.vertex_index + 2] };

				Vec3 normal = normalize(cross(v1 - v0, v2 - v0));

				// 色や放射輝度は適宜設定
				Vec3 color = { 0.8f, 0.3f, 0.3f };
				Vec3 emission = { 0.0f, 0.0f, 0.0f };

				// シーンに三角形を追加
				h_scene.triangles[h_scene.num_triangles++] = { v0, v1, v2, normal, color, emission };

				index_offset += fv;

				obj_bb_max.x = fmaxf(fmaxf(v0.x, v1.x), fmaxf(v2.x, obj_bb_max.x));
				obj_bb_max.y = fmaxf(fmaxf(v0.y, v1.y), fmaxf(v2.y, obj_bb_max.y));
				obj_bb_max.z = fmaxf(fmaxf(v0.z, v1.z), fmaxf(v2.z, obj_bb_max.z));
				obj_bb_min.x = fminf(fminf(v0.x, v1.x), fminf(v2.x, obj_bb_min.x));
				obj_bb_min.y = fminf(fminf(v0.y, v1.y), fminf(v2.y, obj_bb_min.y));
				obj_bb_min.z = fminf(fminf(v0.z, v1.z), fminf(v2.z, obj_bb_min.z));
			}
		}
		std::cout << "obj_bb_min(" << obj_bb_min.x << ", " << obj_bb_min.y << ", " << obj_bb_min.z << ")" << std::endl;
		std::cout << "obj_bb_max(" << obj_bb_max.x << ", " << obj_bb_max.y << ", " << obj_bb_max.z << ")" << std::endl;
	}

	// シーンデータをデバイスメモリにコピー
	Scene* d_scene;
	cudaMallocManaged(&d_scene, sizeof(Scene));
	cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice);
	cudaError_t scene_copy_error = cudaGetLastError();
	if (scene_copy_error != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(scene_copy_error) << std::endl;
		exit(1);
	}


	const float focal_length = 5.0f;
	Vec3 camera_pos = { 20.0f, 7.0f, 30.0f };
	Vec3 look_at_pos = { 0.0f, 0.0f, 0.0f };

	Vec3 camera_up = { 0.0f, 1.0f, 0.0f };
	Camera camera(camera_pos, look_at_pos - camera_pos, camera_up, WIDTH, HEIGHT, nullptr);
	camera.look_at(look_at_pos);
	camera.print_camera();

	dim3 block(16, 16);
	dim3 grid(WIDTH / block.x + 1,
		HEIGHT / block.y + 1);
	std::cout << "grid.x:" << grid.x << " grid.y:" << grid.y << std::endl;

	// create random number generator
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, WIDTH * HEIGHT * sizeof(curandState)));
	render_init << <grid, block >> > (d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	std::cout << "done render_init()" << std::endl;

	std::chrono::time_point<std::chrono::system_clock> start_time = std::chrono::system_clock::now();
	render_pixel << <grid, block >> > (d_output, d_rand_state, d_scene, camera);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_time = end_time - start_time;
	std::cout << "elapsed time:" << elapsed_time.count() << "s" << std::endl << "elapsed time:" << elapsed_time.count() * 1000 / (WIDTH * HEIGHT * RAY_SAMPLES) << "ms/ray" << std::endl;


	float* h_output = new float[WIDTH * HEIGHT * 3];
	cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * 3 * sizeof(float), cudaMemcpyDeviceToHost);


	// output h_output as hdf5
	{
		std::vector<double> h_output_vec(WIDTH * HEIGHT * 3);
		//for (int i = 0; i < WIDTH * HEIGHT * 3; ++i) {
		//	h_output_vec[i] = h_output[i];
		//	// u, value
		//	printf("x:%02d, value:%f -> value:%f\n", i, h_output[i], h_output_vec[i]);
		//}
		for (int y = 0; y < HEIGHT; ++y) {
			for (int x = 0; x < WIDTH; ++x) {
				const int pixel_index = y * WIDTH + x;
				for (int c = 0; c < 3; ++c) {
					h_output_vec[pixel_index * 3 + c] = h_output[pixel_index * 3 + c];
					// u, value
					//printf("x:%02d, y:%02d, c:%02d, value:%f -> value:%f\n", x, y, c, h_output[pixel_index * 3 + c], h_output_vec[pixel_index * 3 + c]);
				}
				//printf("x:%d, y:%d, value(%f, %f, %f)\n", x, y, h_output_vec[pixel_index * 3 + 0], h_output_vec[pixel_index * 3 + 1], h_output_vec[pixel_index * 3 + 2]);
			}
		}
		printf("convert to vector\n");



		Image image = FlattenArray2Image(h_output_vec, WIDTH, HEIGHT, 3);
		std::cout << "image size : " << image.size() << ", " << image[0].size() << std::endl;


		// Open a file
		using HighFive::File;
		File file("tmp.h5", File::ReadWrite | File::Truncate);

		// Create DataSet and write data (short form)
		file.createDataSet("/group/dset1", image);

		// Read the data
		//std::vector<int> d1_read;
		//file.getDataSet("/group/dset1").read(d1_read);
	}


	delete[] h_output;
	cudaFree(d_output);
	cudaFree(d_scene);
	cudaFree(d_rand_state);

	return 0;
}
