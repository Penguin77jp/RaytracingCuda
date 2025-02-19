#pragma once
#include <fstream>
#include <nlohmann/json.hpp>

Vec3 get_vec3_from_json(const nlohmann::json& j, const std::string& attribute_name) {
	return Vec3{
		j[attribute_name][0].get<float>(),
		j[attribute_name][1].get<float>(),
		j[attribute_name][2].get<float>()
	};
}