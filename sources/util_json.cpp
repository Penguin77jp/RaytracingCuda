#include "ray.cuh"
#include "util_json.h"
#include <iostream>
#include <fstream>

Vec3 get_vec3_from_json(const nlohmann::json& j, const std::string& attribute_name) {
	float f3[3];
	for (int i = 0; i < 3; ++i) {
		try {
			f3[i] = j[attribute_name][i].get<float>();
		}
		catch (...) {
			std::cout << "try to get " << attribute_name << "[" << i << "]" << " from json" << std::endl;
			//std::cout << "Error: " << e.what() << std::endl;
			exit(1);
		}
	}

	return Vec3{ f3[0], f3[1], f3[2] };
}

json read_json(const std::string& json_file) {
	std::cout << "loading json : " << "path : " << json_file << std::endl
		<< "abs path : " << std::filesystem::absolute(json_file) << std::endl;
	// check existance json file
	if (!std::filesystem::exists(json_file)) {
		std::cout << "Error: json file is not found" << std::endl;
		exit(1);
	}

	std::ifstream ifs(json_file);
	json data;
	try {
		data = json::parse(ifs);
	}
	catch (json::parse_error& e) {
		std::cout << "Error: json parse error" << std::endl;
		exit(1);
	}
	return data;
}