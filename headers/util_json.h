#pragma once
#include <string>
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

template <typename T>
T get_from_json(const json& j, const std::string& attribute_name) {
	try {
		return j[attribute_name].get<T>();
	}
	catch (...) {
		std::cout << "try to get " << attribute_name << " from json" << std::endl;
		exit(1);
	}
}

Vec3 get_vec3_from_json(const nlohmann::json& j, const std::string& attribute_name);

json read_json(const std::string& json_file);