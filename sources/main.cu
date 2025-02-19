#include <CLI/CLI.hpp>
#include <string>
#include "render.cuh"

#define RAY_SAMPLES 1e3
#define PI 3.14159265358979323846f
#define RAY_EPSILON 1.0e-6f
#define MAX_TRACE_DEPTH 10
#define MAX_TRIANGLES 10000

// Function to print compiler version information.
void print_compiler_version() {
	std::cout << "C++ standard macro (__cplusplus): " << __cplusplus << std::endl;

#ifdef __CUDACC_VER_MAJOR__
	std::cout << "Compiled with nvcc version: "
		<< __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << std::endl;
#elif defined(__VERSION__)
	std::cout << "Compiler version (__VERSION__): " << __VERSION__ << std::endl;
#endif
}

int main(int argc, char* argv[]) {
	print_compiler_version();

	CLI::App app{ "Raytracing CUDA Application" };
	// cameraオプションを定義し、デフォルト値を "camera.json" に設定
	std::string camera_json_file = "";
	app.add_option("--camera", camera_json_file, "Camera JSON file")->default_val("../assets/camera_test.json");
	CLI11_PARSE(app, argc, argv);

	// check cuda init
	checkCudaErrors(cudaDeviceReset());

	//Sphere sphere = { {0.0f, 0.0f, 0.0f}, 1.0f, {0.6f, 0.4f, 0.2f} };
	Scene h_scene;
	h_scene.num_spheres = 0;
	h_scene.num_triangles = 0;
	Camera* camera = nullptr;

	if (false) {
		h_scene.num_spheres = 1;
		h_scene.spheres[0] = { {0.0f, -1000.0f - 10.0f, 0.0f}, 1000.0f, {0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 0.0f} };
	}
	else if (true) {
		h_scene.num_spheres = 4;
		h_scene.spheres[0] = { {0.0f, 0.0f, 0.0f}, 1.0f, {0.6f, 0.4f, 0.2f}, {0.0f, 0.0f, 0.0f} };
		h_scene.spheres[1] = { {2.0f, 0.0f, -1.0f}, 1.0f, {0.2f, 0.4f, 0.6f}, {0.0f, 0.0f, 0.0f} };
		// ground
		h_scene.spheres[2] = { {0.0f, -1000.0f - 10.0f, 0.0f}, 1000.0f, {0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 0.0f} };
		// light
		h_scene.spheres[3] = { {0.0f, 1.5f, 0.0f}, 0.25f, {1.0f, 1.0f, 1.0f}, {10.0f, 10.0f, 10.0f} };
	}
	else if (false) {
		// light and ground
		h_scene.num_spheres = 2;
		//h_scene.spheres[0] = { {0.0f, 0.0f, 0.0f}, 8.0f, {0.0f, 0.0f, 0.0f}, {10.0f, 10.0f, 10.0f} };
		h_scene.spheres[0] = { {0.0f, 3.5f + 5.0f, 2.0f}, 5.0f, {0.0f, 0.0f, 0.0f}, {10.0f, 10.0f, 10.0f} };
		h_scene.spheres[1] = { {0.0f, -1000.0f - 4.0f, 0.0f}, 1000.0f, {0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 0.0f} };

		Vec3 camera_pos = { 0.0f, 0.0f, 10.0f };
		Vec3 look_at_pos = { 0.0f, 0.0f, 0.0f };

		Vec3 camera_up = { 0.0f, 1.0f, 0.0f };
		//camera = new Camera(camera_pos, look_at_pos - camera_pos, camera_up, WIDTH, HEIGHT, nullptr);
		//camera->look_at(look_at_pos);
	}
	else if (true) {
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


		const float focal_length = 5.0f;
		Vec3 camera_pos = { 20.0f, 7.0f, 30.0f };
		Vec3 look_at_pos = { 0.0f, 0.0f, 0.0f };

		Vec3 camera_up = { 0.0f, 1.0f, 0.0f };
		//camera = new Camera(camera_pos, look_at_pos - camera_pos, camera_up, WIDTH, HEIGHT, nullptr);
		//camera->look_at(look_at_pos);
	}
	camera = new Camera(camera_json_file);
	const int width = camera->width;
	const int height = camera->height;

	float* d_output;
	checkCudaErrors(cudaMallocManaged(&d_output, width * height * 3 * sizeof(float)));


	// シーンデータをデバイスメモリにコピー
	Scene* d_scene;
	cudaMallocManaged(&d_scene, sizeof(Scene));
	cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice);
	cudaError_t scene_copy_error = cudaGetLastError();
	if (scene_copy_error != cudaSuccess) {
		std::cerr << "Error: " << cudaGetErrorString(scene_copy_error) << std::endl;
		exit(1);
	}

	dim3 block(16, 16);
	dim3 grid(width / block.x + 1,
		height / block.y + 1);
	std::cout << "grid.x:" << grid.x << " grid.y:" << grid.y << std::endl;

	// create random number generator
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, width * height * sizeof(curandState)));
	render_init << <grid, block >> > (d_rand_state, width, height);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	std::cout << "done render_init()" << std::endl;

	std::chrono::time_point<std::chrono::system_clock> start_time = std::chrono::system_clock::now();
	render_pixel<<<grid, block >>>(d_output, d_rand_state, d_scene, *camera);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_time = end_time - start_time;
	std::cout << "elapsed time:" << elapsed_time.count() << "s" << std::endl << "elapsed time:" << elapsed_time.count() * 1000 / (width * height * RAY_SAMPLES) << "ms/ray" << std::endl;


	float* h_output = new float[width * height * 3];
	cudaMemcpy(h_output, d_output, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost);


	// output h_output as hdf5
	{
		std::vector<double> h_output_vec(width * height * 3);
		//for (int i = 0; i < WIDTH * HEIGHT * 3; ++i) {
		//	h_output_vec[i] = h_output[i];
		//	// u, value
		//	printf("x:%02d, value:%f -> value:%f\n", i, h_output[i], h_output_vec[i]);
		//}
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				const int pixel_index = y * width + x;
				for (int c = 0; c < 3; ++c) {
					h_output_vec[pixel_index * 3 + c] = h_output[pixel_index * 3 + c];
					// u, value
					//printf("x:%02d, y:%02d, c:%02d, value:%f -> value:%f\n", x, y, c, h_output[pixel_index * 3 + c], h_output_vec[pixel_index * 3 + c]);
				}
				//printf("x:%d, y:%d, value(%f, %f, %f)\n", x, y, h_output_vec[pixel_index * 3 + 0], h_output_vec[pixel_index * 3 + 1], h_output_vec[pixel_index * 3 + 2]);
			}
		}
		printf("convert to vector\n");



		Image image = FlattenArray2Image(h_output_vec, width, height, 3);
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
