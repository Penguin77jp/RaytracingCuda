#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#define PI 3.14159265358979323846f


struct Vec3 {
    float x, y, z;
};

inline __host__ __device__
Vec3 operator+(const Vec3& a, const Vec3& b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}
inline __host__ __device__
Vec3 operator-(const Vec3& a, const Vec3& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

inline __host__ __device__
Vec3 operator-(const Vec3& a) {
    return { -a.x, -a.y, -a.z };
}

inline __host__ __device__
Vec3 operator*(const Vec3& a, const float b) {
    return { a.x * b, a.y * b, a.z * b };
}

inline __host__ __device__
Vec3 operator*(const float a, const Vec3& b) {
    return { a * b.x, a * b.y, a * b.z };
}

inline __host__ __device__
Vec3 operator*(const Vec3& a, const Vec3& b) {
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}

inline __host__ __device__
Vec3 operator/(const Vec3& a, const float b) {
    return { a.x / b, a.y / b, a.z / b };
}

inline __host__ __device__
Vec3 operator/(const Vec3& a, const Vec3& b) {
    return { a.x / b.x, a.y / b.y, a.z / b.z };
}

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

inline __host__ __device__
float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__
Vec3 cross(const Vec3& a, const Vec3& b) {
    return { a.y * b.z - a.z * b.y,
             a.z * b.x - a.x * b.z,
             a.x * b.y - a.y * b.x };
}

inline __host__ __device__
float length_squared(const Vec3& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

inline __host__ __device__
float length(const Vec3& v) {
    return sqrtf(length_squared(v));
}

inline __host__ __device__
Vec3 normalize(const Vec3& v) {
    float len = length(v);
    return { v.x / len, v.y / len, v.z / len };
}
