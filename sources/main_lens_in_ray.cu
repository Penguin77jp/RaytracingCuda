#include <CLI/CLI.hpp>
#include <string>
#include "render.cuh"

#define RAY_SAMPLES 1e3
#define RAY_EPSILON 1.0e-6f
#define MAX_TRACE_DEPTH 10
#define MAX_TRIANGLES 10000

int main(int argc, char** argv) {
	CLI::App app("Ray Tracing Renderer");
	std::string input_file = "scene.json";
	app.add_option("-i,--input", input_file, "Input file path")->check(CLI::ExistingFile);
	std::string output_file = "output.h5";
	app.add_option("-o,--output", output_file, "Output file path");
	CLI11_PARSE(app, argc, argv);

	// test lens in ray
	Camera camera("../assets/camera_test.json");



	return 0;
}
