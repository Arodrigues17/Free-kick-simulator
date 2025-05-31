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
const double Cm = 0.005;

const double field_length = 105.0;
const double field_width = 68.0;

const double goal_width = 7.32;
const double goal_height = 2.44;

const double wall_width = 3.0;
const double wall_height = 2.5;

double compute_Cd(double v) {
    return 0.25; // Fixed drag coefficient for football
}

double compute_Cl(double S) {
    const double k1 = 1.2;
    const double k2 = 0.2;
    return (S > 0.0) ? k1 * S / (k2 + S) : 0.0;
}

void compute_forces(const Vec3& vel, const Vec3& omega, Vec3& F_drag, Vec3& F_lift, Vec3& F_buoyancy, Vec3& M_drag) {
    double v_mag = vel.norm();
    double omega_mag = omega.norm();
    double S = (v_mag > 1e-6) ? (R * omega_mag) / v_mag : 0.0;

    double Cd = compute_Cd(v_mag);
    double Cl_eff = compute_Cl(S);

    F_drag = -0.5 * rho * Cd * A * v_mag * vel;

    Vec3 v_cross_omega = vel.cross(omega);
    Vec3 lift_direction = (v_cross_omega.norm() > 1e-6) ? v_cross_omega : Vec3{0.0, 0.0, 0.0};
    lift_direction.normalize();
    F_lift = 0.5 * rho * Cl_eff * A * v_mag * v_mag * lift_direction;

    double V_ball = (4.0 / 3.0) * M_PI * std::pow(R, 3);
    F_buoyancy = {0.0, V_ball * rho * g, 0.0};

    M_drag = {0.0, 0.0, 0.0}; // Ignore spin decay for now
}

bool check_wall_collision(const Vec3& prev_pos, const Vec3& curr_pos, const Vec3& wall_center, const Vec3& to_goal) {
    Vec3 normal = to_goal;
    double d_prev = (prev_pos - wall_center).dot(normal);
    double d_curr = (curr_pos - wall_center).dot(normal);

    if (d_prev * d_curr <= 0.0) {
        double alpha = d_prev / (d_prev - d_curr);
        Vec3 collision_pos = prev_pos + (curr_pos - prev_pos) * alpha;

        if (collision_pos.y < 0.0 || collision_pos.y > wall_height) return false;

        Vec3 right = normal.cross(Vec3{0,1,0});
        double width_offset = (collision_pos - wall_center).dot(right);

        if (std::abs(width_offset) > wall_width / 2.0) return false;

        std::cout << "Ball hit the wall at position: " << collision_pos << "\n";
        return true;
    }

    return false;
}

bool check_goal_collision(const Vec3& pos) {
    return (pos.x <= 0.0) &&
           (pos.z >= (field_width / 2.0) - (goal_width / 2.0) && pos.z <= (field_width / 2.0) + (goal_width / 2.0)) &&
           (pos.y >= 0.0 && pos.y <= goal_height);
}

int main(int argc, char* argv[]) {
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0] << " v0 angle_y direction_angle spin_x spin_y spin_z x0 y0 z0 [output_file]\n";
        return 1;
    }

    double v0 = std::stod(argv[1]);
    double angle_y = std::stod(argv[2]);
    double direction_angle = std::stod(argv[3]);
    Vec3 spin = {std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6])};
    Vec3 start_pos = {std::stod(argv[7]), std::stod(argv[8]), std::stod(argv[9])};
    std::string output_file = (argc >= 11) ? argv[10] : "output/wavepoints.csv";

    double angle_y_rad = angle_y * M_PI / 180.0;
    double direction_rad = direction_angle * M_PI / 180.0;

    // Initial direction vector toward goal
    Vec3 goal_center = {0.0, 0.0, 34.0};
    Vec3 to_goal = goal_center - start_pos;
    to_goal.y = 0;
    to_goal.normalize();

    // Apply horizontal direction rotation (left/right)
    double cos_dir = cos(direction_rad);
    double sin_dir = sin(direction_rad);
    Vec3 to_shot = {
        to_goal.x * cos_dir - to_goal.z * sin_dir,
        0.0,
        to_goal.x * sin_dir + to_goal.z * cos_dir
    };
    to_shot.normalize();

    double v_horizontal = v0 * cos(angle_y_rad);
    Vec3 vel = {v_horizontal * to_shot.x, v0 * sin(angle_y_rad), v_horizontal * to_shot.z};
    Vec3 omega = spin;
    Vec3 pos = start_pos;

    Vec3 wall_center = start_pos + to_shot * 9.15;
    wall_center.y = 0.0;

    const double dt = 0.01;
    const int max_steps = 10000;
    bool hit_wall = false, scored = false;
    std::vector<Vec3> trajectory;
    trajectory.push_back(pos);

    Vec3 prev_pos = pos;
    for (int t = 0; t < max_steps; ++t) {
        Vec3 F_drag, F_lift, F_buoyancy, M_drag;
        compute_forces(vel, omega, F_drag, F_lift, F_buoyancy, M_drag);
        Vec3 F_gravity = {0.0, -m * g, 0.0};

        Vec3 acc = (F_gravity + F_buoyancy + F_drag + F_lift) / m;
        double I = (2.0 / 5.0) * m * R * R;
        Vec3 domega = M_drag / I;

        vel = vel + acc * dt;
        pos = pos + vel * dt;
        omega = omega + domega * dt;

        trajectory.push_back(pos);

        if (check_wall_collision(prev_pos, pos, wall_center, to_shot)) {
            std::cout << "Ball hit the wall at t = " << t * dt << " s.\n";
            hit_wall = true;
            break;
        }
        if (check_goal_collision(pos)) {
            std::cout << "Goal scored at t = " << t * dt << " s!\n";
            scored = true;
            break;
        }
        if (pos.y < 0.0) break;
        if (pos.x < 0.0 || pos.x > field_length || pos.z < 0.0 || pos.z > field_width) break;

        prev_pos = pos;
    }

    if (!hit_wall && scored) {
        std::ofstream csv(output_file);
        csv << "x,y,z,object_type,width,height,orientation_x,orientation_y,orientation_z\n";
        csv << std::fixed << std::setprecision(4);

        for (const auto& p : trajectory) {
            csv << p.x << "," << p.y << "," << p.z << ",ball,1,1,0,0,0\n";
        }

        csv << wall_center.x << "," << wall_center.y << "," << wall_center.z
            << ",wall," << wall_width << "," << wall_height << ","
            << to_goal.x << "," << to_goal.y << "," << to_goal.z << "\n";

        csv << goal_center.x << "," << goal_center.y << "," << goal_center.z
            << ",goal," << goal_width << "," << goal_height << ",1,0,0\n";

        csv.close();
        std::cout << "Trajectory written to " << output_file << "\n";
    } else {
        std::cout << "No trajectory written (ball hit wall or missed goal).\n";
        return 1;
    }

    return 0;
}
