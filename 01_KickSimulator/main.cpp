#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>

struct Vec3 {
    double x, y, z;
};

// Constants for physics
const double g = 9.81;       // Gravity
const double rho = 1.2;      // Air density (kg/m^3)
const double Cd = 0.25;      // Drag coefficient
const double A = 0.038;      // Cross-sectional area (m^2)
const double m = 0.43;       // Mass of soccer ball (kg)
const double Cm = 1.0;       // Magnus coefficient

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(double s, const Vec3& v) {
    return {s * v.x, s * v.y, s * v.z};
}

Vec3 operator*(const Vec3& v, double s) {
    return {s * v.x, s * v.y, s * v.z};
}

Vec3 operator/(const Vec3& v, double s) {
    return {v.x / s, v.y / s, v.z / s};
}

double norm(const Vec3& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Write a sphere by sampling points on a grid (simplified)
void add_sphere(std::vector<Vec3>& points, Vec3 center, double radius, int resolution = 10) {
    for (int i = 0; i <= resolution; ++i) {
        double theta = i * M_PI / resolution;
        for (int j = 0; j <= resolution; ++j) {
            double phi = j * 2 * M_PI / resolution;
            double x = center.x + radius * sin(theta) * cos(phi);
            double y = center.y + radius * sin(theta) * sin(phi);
            double z = center.z + radius * cos(theta);
            points.push_back({x, y, z});
        }
    }
}

void write_vtp(const std::string& filename, const std::vector<Vec3>& points) {
    std::ofstream file(filename);

    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "  <PolyData>\n";
    file << "    <Piece NumberOfPoints=\"" << points.size() << "\" NumberOfVerts=\"1\">\n";
    file << "      <Points>\n";
    file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    for (const auto& pt : points) {
        file << "          " << pt.x << " " << pt.y << " " << pt.z << "\n";
    }

    file << "        </DataArray>\n";
    file << "      </Points>\n";
    file << "      <Verts>\n";
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int i = 0; i < points.size(); ++i) file << i << " ";
    file << "\n        </DataArray>\n";
    file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    file << points.size() << "\n";
    file << "        </DataArray>\n";
    file << "      </Verts>\n";
    file << "    </Piece>\n";
    file << "  </PolyData>\n";
    file << "</VTKFile>\n";

    file.close();
}

int main() {
    const double v0 = 25.0;         // Initial velocity in m/s
    const double angle_deg = 20.0;  // Launch angle in degrees
    const double angle_rad = angle_deg * M_PI / 180.0;
    const double dt = 0.01;         // Time step
    const int num_steps = 200;      // Number of timesteps

    // Initial position and velocity
    Vec3 pos = {0.0, 0.0, 0.0};
    Vec3 vel = {v0 * cos(angle_rad), 0.0, v0 * sin(angle_rad)};
    Vec3 spin = {0.0, 50.0, 0.0}; // Spin around y-axis (Magnus)

    for (int t = 0; t < num_steps; ++t) {
        std::vector<Vec3> ball;
        std::ostringstream filename;

        // Compute forces
        Vec3 drag = -0.5 * rho * Cd * A * norm(vel) * vel / m;
        Vec3 magnus = Cm * cross(spin, vel) / m;
        Vec3 gravity = {0.0, 0.0, -g};
        Vec3 acc = gravity + drag;

        // Integrate
        vel = vel + acc * dt;
        pos = pos + vel * dt;

        // Render
        add_sphere(ball, pos, 0.22);
        filename << "output/ball_t" << t << ".vtp";
        write_vtp(filename.str(), ball);
        std::cout << "t: " << t << ", ax: " << acc.x << ", ay: " << acc.y << ", az: " << acc.z;
        std::cout << ", vx: " << vel.x << ", vy: " << vel.y << ", vz: " << vel.z;
        std::cout << ", px: " << pos.x << ", py: " << pos.y << ", pz: " << pos.z;
        std::cout << std::endl;

        if (pos.z < 0.0) break; // Stop when it hits the ground
    }

    return 0;
}