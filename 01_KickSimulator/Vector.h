#pragma once
#include <cmath>
#include <iostream>

class Vec3 {
public:
    double x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }
    Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }
    Vec3 operator*(double s) const {
        return Vec3(x * s, y * s, z * s);
    }
    friend Vec3 operator*(double s, const Vec3& v) {
        return Vec3(v.x * s, v.y * s, v.z * s);
    }
    Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }
    friend std::ostream& operator<<(std::ostream& os, const Vec3& v) {
        os << "x: " << v.x << ", y: " << v.y << ", z: " << v.z;
        return os;
    }
    double dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }
    Vec3 cross(const Vec3& v) const {
        return Vec3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }
    Vec3 operator/(double s) const {
        if (s == 0) {
            throw std::runtime_error("Division by zero in Vec3.");
        }
        return Vec3(x / s, y / s, z / s);
    }
    void normalize() {
        double len = std::sqrt(x * x + y * y + z * z);
        if (len > 0) {
            x /= len;
            y /= len;
            z /= len;
        }
    }
    double norm() const {
        return std::sqrt(x * x + y * y + z * z);
    }
};
