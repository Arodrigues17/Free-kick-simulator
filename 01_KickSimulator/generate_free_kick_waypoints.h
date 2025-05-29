#pragma once
#include <vector>
#include <random>
#include <cmath>
#include "Vector.h"

inline std::vector<Vector> generate_free_kick_waypoints(int N = 60) {
    // Field and goal parameters
    const double GOAL_X = 0.0;
    const double GOAL_WIDTH = 7.32;
    const double GOAL_HEIGHT = 2.44;
    const double GOAL_Z_LEFT = -GOAL_WIDTH / 2;
    const double GOAL_Z_RIGHT = GOAL_WIDTH / 2;
    
    // Free kick start position
    const double KICK_X = 25.0;
    const double KICK_Y = 0.0;
    const double KICK_Z = 3.0;

    // Random engine
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> ydist(1.0, GOAL_HEIGHT - 1.0);
    std::uniform_real_distribution<> zdist(GOAL_Z_LEFT + 1.0, GOAL_Z_RIGHT - 1.0);

    // Target inside goal (goal line at y=0, ground at z=0)
    double TARGET_X = GOAL_X;
    double TARGET_Y = 0.0; // Goal line
    double TARGET_Z = zdist(gen);

    // Control point for Bezier (arc above ground)
    double CONTROL_X = (KICK_X + TARGET_X) / 2.0;
    double CONTROL_Y = (KICK_Y + TARGET_Y) / 2.0;
    double CONTROL_Z = (KICK_Z + TARGET_Z) / 2.0 + 2.0;

    auto bezier = [](double t, const Vector& p0, const Vector& p1, const Vector& p2) {
        return p0 * ((1-t)*(1-t)) + p1 * (2*(1-t)*t) + p2 * (t*t);
    };

    std::vector<Vector> waypoints;
    for (int i = 0; i < N; ++i) {
        double t = i / double(N-1);
        Vector pos = bezier(t, Vector(KICK_X, KICK_Y, KICK_Z), Vector(CONTROL_X, CONTROL_Y, CONTROL_Z), Vector(TARGET_X, TARGET_Y, TARGET_Z));
        waypoints.push_back(pos);
    }
    return waypoints;
}
