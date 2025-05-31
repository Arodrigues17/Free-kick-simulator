#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include <iomanip>
#include "Vector.h"

// Constants
const double g = 9.81;
const double rho = 1.205;
const double R = 0.11;
const double A = M_PI * R * R;
const double m = 0.43;
const double nu = 1.5e-5;
const double Cd0 = 0.2;
const double Cd_spin = 0.33;
const double Cl = 0.3;
const double Cm = 0.05;

// Field dimensions
const double field_length = 105.0;
const double field_width = 68.0;

// Goal dimensions
const double goal_width = 7.32;
const double goal_height = 2.44;
const double goal_depth = 0.5;

// Wall parameters
const double wall_width = 3.0;
const double wall_height = 2.0;
const double wall_thickness = 0.3;

// Compute forces
void compute_forces(const Vec3& vel, const Vec3& omega, Vec3& F_drag, Vec3& F_lift, Vec3& F_buoyancy, Vec3& M_drag) {
    double v_mag = vel.norm();
    double Re = v_mag * (2 * R) / nu;
    double S = R * omega.norm() / nu;
    double Cd = (S > 0.33) ? Cd_spin : Cd0;
    double Cl_eff = Cl;

    F_drag = -0.5 * rho * Cd * A * v_mag * vel;
    Vec3 v_cross_omega = vel.cross(omega);
    F_lift = (v_cross_omega.norm() > 1e-6) ? 0.5 * rho * Cl_eff * A * v_mag * v_cross_omega : Vec3{0.0, 0.0, 0.0};
    double V_ball = (4.0 / 3.0) * M_PI * std::pow(R, 3);
    F_buoyancy = {0.0, V_ball * rho * g, 0.0};
    M_drag = -0.5 * rho * Cm * A * v_mag * omega;
}   

// Check wall collision
bool check_wall_collision(const Vec3& pos, const Vec3& wall_center) {
    return (pos.x >= wall_center.x - wall_thickness / 2.0 && pos.x <= wall_center.x + wall_thickness / 2.0) &&
           (pos.z >= wall_center.z - wall_width / 2.0 && pos.z <= wall_center.z + wall_width / 2.0) &&
           (pos.y >= 0.0 && pos.y <= wall_height);
}

// Check goal collision
bool check_goal_collision(const Vec3& pos) {
    return (pos.x >= 0.0 && pos.x <= goal_depth) &&
           (pos.z >= (field_width / 2.0) - (goal_width / 2.0) && pos.z <= (field_width / 2.0) + (goal_width / 2.0)) &&
           (pos.y >= 0.0 && pos.y <= goal_height);
}

int main() {
    // Ball initial conditions
    double v0 = 40.0;               // Initial speed (m/s)
    double angle_y = 25.0;          // Vertical launch angle (degrees)
    double angle_y_rad = angle_y * M_PI / 180.0;
    Vec3 spin = {2.0, 0.0, -9.0};   // Side spin

    // Ball starts 30m from goal at center of pitch width
    Vec3 pos = {30.0, 0.0, 34.0};  // (x, y, z)

    // Compute horizontal direction from ball to goal center
    Vec3 goal_center = {0.0, 0.0, 34.0};
    Vec3 to_goal = goal_center - pos;
    to_goal.y = 0.0;
    Vec3 dir_xz = to_goal;
    dir_xz.normalize();  // Normalize to get direction vector
    dir_xz.z = -0.1;

    std::cout << "to_goal: " << to_goal << std::endl;
    std::cout << "dir_xz: " << dir_xz << std::endl;


    // Compute velocity
    double v_horizontal = v0 * cos(angle_y_rad);
    Vec3 vel = {
        v_horizontal * dir_xz.x,     // x component
        v0 * sin(angle_y_rad),       // y (height)
        v_horizontal * dir_xz.z      // z component
    };

    Vec3 omega = spin;

    // Wall at 9.15m from goal, centered
    Vec3 wall_center = {pos.x - 9.15, 1.0, 34.0};

    // Goal center
    Vec3 goal_center_obj = {0.25, 1.22, 34.0};

    // CSV output
    std::ofstream csv("output/wavepoints.csv");
    csv << "x,y,z,object_type,width,length\n";
    csv << std::fixed << std::setprecision(4);

    // Write wall and goal to CSV
    csv << wall_center.x << "," << wall_center.y << "," << wall_center.z << ",wall," << 3.0 << "," << 2.0 << "\n";
    csv << goal_center_obj.x << "," << goal_center_obj.y << "," << goal_center_obj.z << ",goal," << 7.32 << "," << 2.44 << "\n";

    // Simulation loop
    const double dt = 0.01;
    const int max_steps = 10000;
    bool hit_wall = false;
    bool scored = false;

    for (int t = 0; t < max_steps; ++t) {
        Vec3 F_drag, F_lift, F_buoyancy, M_drag;
        compute_forces(vel, omega, F_drag, F_lift, F_buoyancy, M_drag);
        Vec3 F_gravity = {0.0, -m * g, 0.0};

        Vec3 acc = (F_gravity + F_buoyancy + F_drag + F_lift) / m;
        double I = (2.0 / 5.0) * m * R * R;
        Vec3 domega = M_drag / I;

        vel = vel + acc * dt;
        std::cout << "Velocity: " << vel << "\n";
        pos = pos + vel * dt;
        omega = omega + domega * dt;

        csv << pos.x << "," << pos.y << "," << pos.z << ",ball,1,1\n";

        if (check_wall_collision(pos, wall_center)) {
            std::cout << "Ball hit the wall at t = " << t * dt << " s.\n";
            hit_wall = true;
            break;
        }
        if (check_goal_collision(pos)) {
            std::cout << "Goal scored at t = " << t * dt << " s!\n";
            scored = true;
            break;
        }
        if (pos.y < 0.0) {
            std::cout << "Ball hit the ground at t = " << t * dt << " s.\n";
            break;
        }
        if (pos.x < 0.0 || pos.x > 105.0 || pos.z < 0.0 || pos.z > 68.0) {
            std::cout << "Ball out of bounds at t = " << t * dt << " s.\n";
            break;
        }
    }

    csv.close();
    if (!hit_wall && !scored) std::cout << "Ball missed the goal.\n";

    return 0;
}
