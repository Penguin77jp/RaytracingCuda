#include "camera.cuh"
#include "util_json.h"

LensSystem::LensSystem(const std::string& json_file, const Camera& cam)
	:object_focal_length(cam.focal_length)
{
	const json json_data = read_json(json_file);
	this->num_lenses = get_from_json<int>(json_data, "num");
	this->focal_length = get_from_json<float>(json_data, "focal_length");
	this->front_principal_plane = get_from_json<float>(json_data, "fonrt_principal_planes");
	this->back_principal_plane = get_from_json<float>(json_data, "back_principal_planes");

	if (LENS_SYSTEM_MAX_LENSES < this->num_lenses) {
		std::cout << "Error: you should more allocate memory for lenses" << std::endl;
		exit(1);
	}

	for (size_t index = 0; const auto & item : json_data["lens"]) {
		const float thickness = get_from_json<float>(item, "thickness");
		const float radius = get_from_json<float>(item, "radius");
		const float diameter = get_from_json<float>(item, "diameter");
		const float refractive_index = get_from_json<float>(item, "refractive_index");

		this->lenses[index] = Lens(radius, diameter, thickness, refractive_index);
		++index;
	}

	// calculate distance to image plane
	{
		const float H = this->front_principal_plane;
		const float H_dash = this->back_principal_plane;
		const float& d = this->object_focal_length;
		this->distance_to_image_plane = H_dash + focal_length * (d - H) / (d - H - focal_length);
		/*
		std::cout << "H_hash : " << H_dash << std::endl;
		std::cout << "focal_length * (d - H) : " << focal_length * (d - H) << std::endl;
		std::cout << "(d - H) : " << (d - H) << std::endl;
		std::cout << "d : " << d << std::endl;
		std::cout << "H : " << H << std::endl;
		std::cout << "(d - H - focal_length) : " << (d - H - focal_length) << std::endl;
		*/
		std::cout << "distance_to_image_plane: " << this->distance_to_image_plane << std::endl;
	}

	this->print();
}

void LensSystem::print() const {
	printf("LensSystem: num_lenses: %d, focal_length: %f, distance_to_image_plane: %f\n", this->num_lenses, this->focal_length, this->distance_to_image_plane);
	for (int i = 0; i < this->num_lenses; i++) {
		printf("Lens %d: radius: %f, diameter: %f, thickness: %f, refractive_index: %f\n", i, this->lenses[i].radius, this->lenses[i].diameter, this->lenses[i].thickness, this->lenses[i].refractive_index);
	}
}


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


	this->width = get_from_json<int>(data, "width");
	this->height = get_from_json<int>(data, "height");
	std::cout << "lens_system : is nullptr" << (this->lens_system.valid()) << std::endl;
	this->focal_length = get_from_json<float>(data, "focal_length");

	// load lens system file
	{
		std::string lens_system_file = "";
		if (data.contains("lens_system")) {
			lens_system_file = get_from_json<std::string>(data, "lens_system");
			const auto base_path = std::filesystem::path(json_file).parent_path();
			const auto lens_system_file_path = base_path / lens_system_file;
			lens_system = LensSystem(lens_system_file_path.string(), *this);
		}
	}

	this->print_camera();
}


