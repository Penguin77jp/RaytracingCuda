#include "camera.cuh"
#include "util_json.h"

Camera::Camera(const std::string& json_file) {
	json data = read_json(json_file);
	this->position = get_vec3_from_json(data, "position");

	// load direction
	{
		if (data.contains("direction")) {
			this->direction = normalize(get_vec3_from_json(data, "direction"));
		}
		else if (data.contains("look_at")) {
			Vec3 look_at = get_vec3_from_json(data, "look_at");
			this->direction = normalize(look_at - this->position);
		}
		else {
			std::cout << "Error: direction or look_at is not found in camera json file" << std::endl;
			exit(1);
		}
	}

	this->up = Vec3{ 0.0f, 1.0f, 0.0f };


	// load lens system file
	{
		std::string lens_system_file = "";
		if (data.contains("lens_system")) {
			lens_system_file = data["lens_system"];
			lens_system = new LensSystem(lens_system_file);
		}
	}

	this->width = get_from_json<int>(data, "width");
	this->height = get_from_json<int>(data, "height");
	std::cout << "lens_system : " << this->lens_system << ", is nullptr" << (this->lens_system == nullptr) << std::endl;
	if (this->lens_system == nullptr) {
		this->focal_length = get_from_json<float>(data, "focal_length");
	}
	else {
		this->focal_length = lens_system->compute_focal_length();
	}

	this->print_camera();
}


