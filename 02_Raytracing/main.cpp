#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <stack>
#include "omp.h"
#include <iomanip>
#include <immintrin.h>
#include <fstream>
#include <sstream> // Required for std::stringstream
#include <string>  // Required for std::string, std::stod
#include <algorithm>
#include <limits> // Required for std::numeric_limits
#include <filesystem> // Added for directory operations
#include "../01_KickSimulator/Vector.h" // Defines Vec3
// #include "../01_KickSimulator/generate_free_kick_waypoints.h" // Removed

using namespace std;

#define EPS 1e-4
int MAX_DEPTH = 5;

enum ANTI_ALIASING {
    AA_NONE,
    AA_RANDOM_2,
    AA_REGULAR_2,
    AA_RANDOM_4,
    AA_REGULAR_4
};

class RGB {
public:
    int r, g, b;

    RGB() : r(0), g(0), b(0) {}
    RGB(int _r, int _g, int _b) : r(_r), g(_g), b(_b) {}

    // Addition
    RGB operator+(const RGB& other) const {
        return RGB(r + other.r, g + other.g, b + other.b);
    }

    RGB& operator+=(const RGB& other) {
        r += other.r; g += other.g; b += other.b;
        return *this;
    }

    // Subtraction
    RGB operator-(const RGB& other) const {
        return RGB(r - other.r, g - other.g, b - other.b);
    }

    RGB& operator-=(const RGB& other) {
        r -= other.r; g -= other.g; b -= other.b;
        return *this;
    }

    // Component-wise multiplication (RGB * RGB)
    RGB operator*(const RGB& other) const {
        return RGB(r * other.r, g * other.g, b * other.b);
    }

    RGB& operator*=(const RGB& other) {
        r *= other.r; g *= other.g; b *= other.b;
        return *this;
    }

    // Scalar multiplication
    RGB operator*(double scalar) const {
        return RGB(r * scalar, g * scalar, b * scalar);
    }

    RGB& operator*=(double scalar) {
        r *= scalar; g *= scalar; b *= scalar;
        return *this;
    }

    // Scalar division
    RGB operator/(double scalar) const {
        return (scalar != 0) ? RGB(r / scalar, g / scalar, b / scalar) : RGB();
    }

    RGB& operator/=(double scalar) {
        if (scalar != 0) {
            r /= scalar; g /= scalar; b /= scalar;
        }
        return *this;
    }

    // Equality and Inequality
    bool operator==(const RGB& other) const {
        return (r == other.r && g == other.g && b == other.b);
    }

    bool operator!=(const RGB& other) const {
        return !(*this == other);
    }

    void clamp() {
        r = min(255, max(0, r));
        g = min(255, max(0, g));
        b = min(255, max(0, b));
    }

    // Output stream overload
    friend std::ostream& operator<<(std::ostream& os, const RGB& color) {
        os << "RGB(" << color.r << ", " << color.g << ", " << color.b << ")";
        return os;
    }
};

class LightSource {
public:
    Vec3 position;
    RGB intensity;

    LightSource(Vec3 _position, RGB _intensity) : position(_position), intensity(_intensity) {}
};

class Ray {
public:
    Vec3 origin;
    Vec3 direction;
    Ray(Vec3 _o, Vec3 _d) : origin(_o), direction(_d) {}
};

// Rodrigues' rotation formula function
Vec3 rotate(Vec3 p, Vec3 axis, double angle) {
    axis.normalize();
    return p * cos(angle) + axis.cross(p) * sin(angle) + 
           axis * (axis.dot(p)) * (1 - cos(angle));
}

class Pixel {
public:
    int x, y;
    RGB color;

    Pixel(int _x, int _y, RGB _color = RGB()) : x(_x), y(_y), color(_color) {}
};

class Material {
public:
    virtual RGB getObjectColor() const = 0;
    virtual double getAmbientCoefficient() const = 0;
    virtual double getDiffusionCoefficient() const = 0;
    virtual double getSpecularCoefficient() const = 0;
    virtual double getShininessCoefficient() const = 0;
    virtual double getRefractionIndex() const = 0;
    virtual ~Material() {}
};

class Shape {
public:
    virtual Vec3 getSurfaceNormal(Vec3& pointOnSurface) = 0;
    virtual double solveRoot(Ray ray) = 0;
    virtual Material* getMaterial() const = 0;
    virtual ~Shape() {}
};

class Sphere : public Shape {
public:
    Vec3 center;
    double R;
    Material* material;
    Vec3 currentRotationAxis; // Added for spin
    double currentRotationAngle; // Added for spin

    Sphere(Vec3 _center, double _R, Material* _material) : center(_center), R(_R), material(_material), currentRotationAxis(0,0,1), currentRotationAngle(0.0) {}

    Material* getMaterial() const override { return material; }

    Vec3 getSurfaceNormal(Vec3& pointOnSurface) override {
        return pointOnSurface - center;
    }

    double solveRoot(Ray ray) override {
        Vec3 oc = ray.origin - center;
        double a = ray.direction.dot(ray.direction);
        double b = 2.0 * oc.dot(ray.direction);
        double c = oc.dot(oc) - R * R;
        double discriminant = b * b - 4 * a * c;

        if (discriminant < 0) {
            return -std::numeric_limits<double>::min();  // No intersection
        }

        double inv2a = 0.5 / a;  // Precompute reciprocal to avoid division
        double sqrtD = sqrt(discriminant);
        double t1 = (-b - sqrtD) * inv2a;
        double t2 = (-b + sqrtD) * inv2a;

        
        if (t1 > 0) {
            if (t2 > 0) return (t1 < t2) ? t1 : t2;
            return t1;
        } 
        return (t2 > 0) ? t2 : -std::numeric_limits<double>::min();        
    }
};

// Define the quartic function and its derivative
double quarticFunction(double t, double a, double b, double c, double d, double e) {
    return a * t * t * t * t + b * t * t * t + c * t * t + d * t + e;
}

double quarticDerivative(double t, double a, double b, double c, double d) {
    return 4 * a * t * t * t + 3 * b * t * t + 2 * c * t + d;
}

// Newton's method for finding real roots of the quartic function
double newtonSolveQuartic(double a, double b, double c, double d, double e, double initialGuess, int maxIterations = 50, double tolerance = 1e-6) {
    double t = initialGuess;
    
    for (int i = 0; i < maxIterations; i++) {
        double f_t = quarticFunction(t, a, b, c, d, e);
        double f_prime_t = quarticDerivative(t, a, b, c, d);

        if (std::abs(f_prime_t) < tolerance) {
            break; // Avoid division by zero
        }

        double t_next = t - f_t / f_prime_t;
        
        if (std::abs(t_next - t) < tolerance) {
            return t_next; // Converged to a root
        }
        
        t = t_next;
    }

    return std::numeric_limits<double>::max(); // If no root was found
}

class Torus : public Shape {
public:
    Vec3 center;
    double R; // Major radius
    double r; // Minor radius
    Material* material;

    Torus(Vec3 _center, double _R, double _r, Material* _material) : center(_center), R(_R), r(_r), material(_material) {}

    Material* getMaterial() const override { return material; }

    Vec3 getSurfaceNormal(Vec3& pointOnSurface) override {
        Vec3 local = pointOnSurface - center;
        double phi = atan2(local.y, local.x);
        Vec3 closestPoint = Vec3(R * cos(phi), R * sin(phi), 0);
        Vec3 normal = (pointOnSurface - (center + closestPoint));
        normal.normalize();
        return normal;
    }

    double solveRoot(Ray ray) override {
        // Transform ray into torus coordinate system
        Vec3 o = ray.origin - center;
        Vec3 d = ray.direction;

        double dDotD = d.dot(d);
        double oDotD = o.dot(d);
        double oDotO = o.dot(o);
        double sumR = R * R + r * r;

        // Quartic equation coefficients
        double a = dDotD * dDotD;
        double b = 4.0 * dDotD * oDotD;
        double c = 4.0 * oDotD * oDotD + 2.0 * dDotD * (oDotO - sumR) + 4.0 * R * R * d.z * d.z;
        double d_coeff = 4.0 * (oDotO - sumR) * oDotD + 8.0 * R * R * o.z * d.z;
        double e = (oDotO - sumR) * (oDotO - sumR) - 4.0 * R * R * (r * r - o.z * o.z);

        // Use Newton's method to solve
        double initialGuess = 1.0;
        double root = newtonSolveQuartic(a, b, c, d_coeff, e, initialGuess);

        if (root == std::numeric_limits<double>::max() || root < 0) {
            return -std::numeric_limits<double>::min();
        }

        return root;
    }
};

class Plane : public Shape {
public:
    Vec3 point;    // A point on the plane (e.g., the origin point on the floor)
    Vec3 normal;   // The normal vector of the plane
    Material* material; // Material for the plane

    Plane(Vec3 _point, Vec3 _normal, Material* _material) : point(_point), normal(_normal), material(_material) {}

    Material* getMaterial() const override { return material; }

    Vec3 getSurfaceNormal(Vec3& pointOnSurface) override {
        return normal; // The normal is constant for a plane
    }

    double solveRoot(Ray ray) override {
        double denominator = 1/ray.direction.dot(normal);
        
        if (fabs(denominator) < EPS) {
            return -std::numeric_limits<double>::max();  // No intersection, ray is parallel to the plane
        }
        
        double t = (point - ray.origin).dot(normal) * denominator;
        
        if (t > EPS) {
            return t;
        }
        
        return -std::numeric_limits<double>::max();  // No valid intersection
    }
};

class KleinBottle : public Shape {
public:
    Vec3 center;
    double R; // Major radius
    double r; // Tube radius
    Material* material;

    KleinBottle(Vec3 _center, double _R, double _r, Material* _material)
        : center(_center), R(_R), r(_r), material(_material) {}

    Material* getMaterial() const override { return material; }

    Vec3 parametric(double u, double v) const {
        double x = (R + cos(u / 2) * sin(v) - sin(u / 2) * sin(2 * v)) * cos(u);
        double y = (R + cos(u / 2) * sin(v) - sin(u / 2) * sin(2 * v)) * sin(u);
        double z = sin(u / 2) * sin(v) + cos(u / 2) * sin(2 * v);
        return center + Vec3(x, y, z);
    }

    Vec3 getSurfaceNormal(Vec3& pointOnSurface) override {
        double u = atan2(pointOnSurface.y - center.y, pointOnSurface.x - center.x);
        double v = atan2(pointOnSurface.z - center.z, sqrt(pow(pointOnSurface.x - center.x, 2) + pow(pointOnSurface.y - center.y, 2)));

        double epsilon = 1e-8;
        Vec3 du = (parametric(u + epsilon, v) - parametric(u - epsilon, v)) * (0.5 / epsilon);
        Vec3 dv = (parametric(u, v + epsilon) - parametric(u, v - epsilon)) * (0.5 / epsilon);
        Vec3 normal = du.cross(dv);
        normal.normalize();

        return normal;
    }

    double solveRoot(Ray ray) override {
        const int maxIterations = 100;
        const double tolerance = 1e-4;
        double t = 1.0;
    
        for (int i = 0; i < maxIterations; i++) {
            Vec3 p = ray.origin + ray.direction * t;
            double u = atan2(p.y - center.y, p.x - center.x);
            double v = atan2(p.z - center.z, sqrt(pow(p.x - center.x, 2) + pow(p.y - center.y, 2)));
            Vec3 q = parametric(u, v);
            Vec3 normal = getSurfaceNormal(q);
    
            double error = (q - p).norm();
            if (error < tolerance) {
                return t;
            }
    
            // Compute ray-plane intersection step
            double denom = normal.dot(ray.direction);
            if (fabs(denom) < 1e-6) { // Avoid division by zero (ray nearly parallel to surface)
                return -std::numeric_limits<double>::min();
            }
    
            double step = normal.dot(q - p) / denom;
            t += step;
    
            if (t < 0) return -std::numeric_limits<double>::min();
        }
        if (t < 1) return t;
        return -std::numeric_limits<double>::min();
    }    
};

class Triangle : public Shape {
public:
    Vec3 v1;
    Vec3 v2;
    Vec3 v3;
    Vec3 normal;   // The normal vector of the plane
    Vec3 n1;
    Vec3 n2;
    Vec3 n3;
    Material* material; // Material for the plane

    Triangle(Vec3 _v1, Vec3 _v2, Vec3 _v3, Material* _material) : v1(_v1), v2(_v2), v3(_v3), material(_material) {
        normal = (v2 - v1).cross(v3 - v1);
        normal.normalize();
    }

    Material* getMaterial() const override { return material; }

    Vec3 getSurfaceNormal(Vec3& pointOnSurface) override {
        // Compute barycentric coordinates
        Vec3 u = v2 - v1;
        Vec3 v = v3 - v1;
        Vec3 w = pointOnSurface - v1;
    
        double d00 = u.dot(u);
        double d01 = u.dot(v);
        double d11 = v.dot(v);
        double d20 = w.dot(u);
        double d21 = w.dot(v);
    
        double denom = d00 * d11 - d01 * d01;
        if (fabs(denom) < EPS) return normal;
    
        double beta = (d11 * d20 - d01 * d21) / denom;
        double gamma = (d00 * d21 - d01 * d20) / denom;
        double alpha = 1.0 - beta - gamma;

        if (alpha < 0 || beta < 0 || gamma < 0 || std::isnan(alpha) || std::isnan(beta) || std::isnan(gamma)) { // Added NaN checks
            return normal;  // fallback to face normal
        }
    
        Vec3 interpolatedNormal = n1 * alpha + n2 * beta + n3 * gamma;
        interpolatedNormal.normalize();
        return interpolatedNormal;
    }

    __attribute__((hot)) double solveRoot(Ray ray) override {
        double denominator = ray.direction.dot(normal);
        if (fabs(denominator) < EPS) {
            return -std::numeric_limits<double>::max();
        }

        denominator = 1/denominator;
    
        double t = (v1 - ray.origin).dot(normal) * denominator;
        if (t <= EPS) {
            return -std::numeric_limits<double>::max();
        }
    
        Vec3 Rp = ray.origin + ray.direction * t;
    
        Vec3 edge1 = v2 - v1;
        Vec3 edge2 = v3 - v2;
        Vec3 edge3 = v1 - v3;
    
        Vec3 C1 = Rp - v1;
        Vec3 C2 = Rp - v2;
        Vec3 C3 = Rp - v3;
    
        double cross1 = edge1.cross(C1).dot(normal);
        if (cross1 < -EPS) return -std::numeric_limits<double>::max();
    
        double cross2 = edge2.cross(C2).dot(normal);
        if (cross2 < -EPS) return -std::numeric_limits<double>::max();
    
        double cross3 = edge3.cross(C3).dot(normal);
        if (cross3 < -EPS) return -std::numeric_limits<double>::max();
    
        return t;
    }
};

class Screen {
public:
    Vec3 normal;
    Vec3 right;
    Vec3 up;
    int width;
    int height;
    double pov;
    std::vector<Pixel> pixels;

    Screen(Vec3 _n, int _w, int _h) : normal(_n), width(_w), height(_h) {
        pixels.reserve(width * height);

        pov = normal.norm(); // 'normal' is the constructor argument _n via member initialization

        // Initialize pixels with default values
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                pixels.emplace_back(x, y, RGB(0, 0, 0));
            }
        }

        // Stabilize camera orientation by defining 'up' relative to global Z-axis
        // First, ensure this->normal is the normalized view direction.
        // The constructor argument _n is initially assigned to the member 'this->normal'.
        if (this->pov > EPS) {
            this->normal = this->normal / this->pov; // Normalize the member 'normal'
        } else {
            this->normal = Vec3(1,0,0); // Default view direction if _n was zero
            if (this->pov < EPS) this->pov = 1.0; // Default pov if _n was zero vector
        }

        Vec3 globalUpDirection(0, 0, 1); // World Z-axis is considered 'up'

        // Calculate screen's right vector (camera's local X-axis)
        // right = cross(globalUp, viewDirection)
        this->right = globalUpDirection.cross(this->normal);
        if (this->right.norm() < EPS) {
            // Camera's view direction (this->normal) is parallel to globalUpDirection.
            // (e.g., looking straight up or down along Z-axis).
            // In this case, define 'right' along the global X-axis.
            // If this->normal is (0,0,1) or (0,0,-1), Vec3(1,0,0) is orthogonal.
            this->right = Vec3(1, 0, 0);
            // If view direction is (0,0,1), right becomes (1,0,0). up = (0,0,1).cross(1,0,0) = (0,-1,0)
            // If view direction is (0,0,-1), right becomes (1,0,0). up = (0,0,-1).cross(1,0,0) = (0,1,0)
            // To ensure up is (0,1,0) when looking down Z, and (0,-1,0) when looking up Z (consistent with right-handed system if X is right, Y is up, Z is forward)
            // we might need to adjust based on the sign of this->normal.z
            if (this->normal.z > 0) { // Looking up along Z
                 this->up = Vec3(0, -1, 0); // up should be -Y global
            } else { // Looking down along Z or other cases where right was (1,0,0)
                 this->up = Vec3(0, 1, 0); // up should be +Y global
            }
            this->up.normalize(); // Should already be normalized
            return; // Skip the rest of the calculation as up and right are set

        } else {
            this->right.normalize();
        }

        // Calculate screen's up vector (camera's local Y-axis)
        // up = cross(viewDirection, right)
        this->up = this->normal.cross(this->right);
        this->up.normalize(); // this->up should already be normalized if this->normal and this->right are ortho-normalized.
        // The vectors (this->right, this->up, this->normal) now form a stable orthonormal basis.
    }

    std::vector<Pixel>::iterator begin() { return pixels.begin(); }
    std::vector<Pixel>::iterator end() { return pixels.end(); }

    void writeToJPG(const std::string& filename, int quality = 90) {
        std::vector<unsigned char> imageData(width * height * 3); // RGB format

        // Convert grayscale pixel data to RGB
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = (y*width + x) * 3;  // RGB index
                RGB pixelColor = pixels[y*width + x].color;
                unsigned char red = static_cast<unsigned char>(pixelColor.r);
                unsigned char green = static_cast<unsigned char>(pixelColor.g);
                unsigned char blue = static_cast<unsigned char>(pixelColor.b);

                // Assign to all RGB channels
                imageData[index] = red;     // R
                imageData[index + 1] = green; // G
                imageData[index + 2] = blue; // B
            }
        }

        // Write the image to a JPG file
        stbi_write_jpg(filename.c_str(), width, height, 3, imageData.data(), quality);
    }
};


class Metallic : public Material {
public:
    // Members
    RGB color;
    double mA = 0.0005;
    double mD = 0.005;
    double mS = 1.0;
    double mSp = 50;
    double eta = 999999;

    // Constructor
    Metallic(RGB _color) : color(_color) {}

    // Getters
    RGB getObjectColor() const override { return color; }
    double getAmbientCoefficient() const override { return mA; }
    double getDiffusionCoefficient() const override { return mD; }
    double getSpecularCoefficient() const override { return mS; }
    double getShininessCoefficient() const override { return mSp; }
    double getRefractionIndex() const override { return eta; }
};

class Glassy : public Material {
public:
    // Members
    RGB color;
    double mA = 0.00001;
    double mD = 0.00001;
    double mS = 0.5;
    double mSp = 300;
    double eta = 1.5;

    // Constructor
    Glassy(RGB _color) : color(_color) {}

    // Getters
    RGB getObjectColor() const override { return color; }
    double getAmbientCoefficient() const override { return mA; }
    double getDiffusionCoefficient() const override { return mD; }
    double getSpecularCoefficient() const override { return mS; }
    double getShininessCoefficient() const override { return mSp; }
    double getRefractionIndex() const override { return eta; }
};

class CheckerboardMaterial : public Material {
public:
    RGB color1, color2;  // Two colors for the checkerboard pattern
    double squareSize;    // Size of each square on the checkerboard

    CheckerboardMaterial(RGB _color1, RGB _color2, double _squareSize)
        : color1(_color1), color2(_color2), squareSize(_squareSize) {}

    RGB getObjectColor() const override { return RGB(255, 255, 255); } // Default to white for shading

    double getAmbientCoefficient() const override { return 1.0; }
    double getDiffusionCoefficient() const override { return 1.0; }
    double getSpecularCoefficient() const override { return 0.0; }
    double getShininessCoefficient() const override { return 0.0; }
    double getRefractionIndex() const override { return 1.0; }

    // Determine the color at a given point on the checkerboard
    RGB getColorAtPoint(const Vec3& point) const {
        int xIndex = abs(static_cast<int>(floor(point.x / squareSize))) % 2;
        int yIndex = abs(static_cast<int>(floor(point.y / squareSize))) % 2;
        // Alternate colors based on the coordinates
        return (xIndex == yIndex) ? color1 : color2;
    }
};

class SoccerBallMaterial : public Material {
public:
    RGB pentagonColor;  // Color for pentagonal faces (traditionally black)
    RGB hexagonColor;   // Color for hexagonal faces (traditionally white)
    double mA, mD, mS, mSp, eta;
    
    SoccerBallMaterial(RGB _pentagonColor, RGB _hexagonColor) 
        : pentagonColor(_pentagonColor), hexagonColor(_hexagonColor),
          mA(0.15), mD(0.85), mS(0.6), mSp(25.0), eta(1.0) {}  // Enhanced material properties

    RGB getObjectColor() const override { return RGB(255, 255, 255); } // Will be overridden
    double getAmbientCoefficient() const override { return mA; }
    double getDiffusionCoefficient() const override { return mD; }
    double getSpecularCoefficient() const override { return mS; }
    double getShininessCoefficient() const override { return mSp; }
    double getRefractionIndex() const override { return eta; }

    // Improved soccer ball pattern that creates a more realistic appearance
    RGB getColorAtPoint(const Vec3& point, const Vec3& sphereCenter) const {
        Vec3 localPoint = point - sphereCenter;
        localPoint.normalize(); // Point on unit sphere centered at origin

        // Convert to spherical coordinates
        double theta = atan2(localPoint.y, localPoint.x); // Azimuthal angle [-π, π]
        double phi = acos(localPoint.z);                   // Polar angle [0, π]
        
        double u = (theta + M_PI) / (2.0 * M_PI);  // [0, 1]
        double v = phi / M_PI;                      // [0, 1]
        
        // Enhanced pattern generation for more distinct features

        // Pentagon pattern (dark patches) - sharper and more defined
        double pentagon_scale_factor = 4.0; // Original scale
        double p_u_norm = fmod(u * pentagon_scale_factor, 1.0);
        double p_v_norm = fmod(v * pentagon_scale_factor, 1.0);
        double pentagon_component = 0.5 + 0.5 * cos(p_u_norm * 2.0 * M_PI * 2.0) * cos(p_v_norm * 2.0 * M_PI * 2.0);
        pentagon_component = pow(pentagon_component, 3.5); // Increased sharpness (original was 2.0)

        // Hexagon pattern (background filler) - slightly sharper
        double hexagon_scale_factor = 6.0; // Original scale
        double h_u_norm = fmod(u * hexagon_scale_factor + 0.5, 1.0);
        double h_v_norm = fmod(v * hexagon_scale_factor + 0.25, 1.0);
        double hexagon_component = 0.5 + 0.5 * cos(h_u_norm * 2.0 * M_PI * 3.0) * cos(h_v_norm * 2.0 * M_PI * 3.0);
        hexagon_component = pow(hexagon_component, 2.0); // Increased sharpness (original was 1.5)
        
        // Noise component - slightly reduced influence
        double noise_factor = 20.0; // Original scale
        double noise_value = 0.5 + 0.5 * sin(u * noise_factor) * cos(v * noise_factor * 1.3);
        
        // Seam lines - make them more pronounced
        double seam_u_param = fmod(u * 10.0, 1.0); // Slightly reduced frequency from 12.0
        double seam_v_param = fmod(v * 7.0, 1.0);  // Slightly reduced frequency from 8.0
        double seam_sharpness_coeff = 75.0; // Original was 50.0, increased for sharper seams
        double seam_mask = 1.0 - exp(-seam_sharpness_coeff * std::min(seam_u_param, 1.0 - seam_u_param) * std::min(seam_v_param, 1.0 - seam_v_param));
        // seam_mask is close to 0 on seams, 1 away from seams.

        // Combine components: Adjust weights for better contrast
        double pattern_mix = pentagon_component * 0.70 + hexagon_component * 0.25 + noise_value * 0.05; // Pentagons more weight, noise less
        
        // Apply seam mask: seams should make the pattern dark.
        pattern_mix *= seam_mask; 
        
        // Latitude variation - slightly reduce its impact
        double lat_var_component = 0.5 + 0.25 * sin(phi * 3.0) * cos(theta * 2.0); // Slightly reduced amplitude from 0.3
        // Modulate overall brightness slightly by latitude variation, ensure base is high enough
        pattern_mix *= (0.9 + lat_var_component * 0.2); // Original was (0.5 + 0.3 * sin * cos) directly multiplying.

        // Threshold for black vs white
        // A lower pattern_mix value should result in pentagonColor (dark)
        if (pattern_mix > 0.38) { // Adjusted threshold (original was 0.45)
            return hexagonColor; 
        } else {
            return pentagonColor; 
        }
    }
};

class SoccerBallSphere : public Sphere {
public:
    SoccerBallSphere(Vec3 _center, double _R, SoccerBallMaterial* _material) 
        : Sphere(_center, _R, _material) {}
    
    Material* getMaterial() const override { return material; }
    
    RGB getColorAtPoint(const Vec3& point) const {
        SoccerBallMaterial* soccerMat = dynamic_cast<SoccerBallMaterial*>(material);
        
        if (soccerMat) {
            Vec3 rotatedPoint = point;
            // Apply rotation if there's spin
            if (currentRotationAngle > EPS && currentRotationAxis.norm() > EPS) {
                Vec3 localPoint = point - center;
                localPoint = rotate(localPoint, currentRotationAxis, -currentRotationAngle);
                rotatedPoint = center + localPoint;
            }
            return soccerMat->getColorAtPoint(rotatedPoint, center);
        }
        
        return RGB(255, 255, 255); // Fallback
    }
};

enum ReflectionMethod {
    FRESNEL,
    SCHLICK
};

RGB traceRay(Ray& ray, Screen& screen, vector<Shape*>& objects, int depth, LightSource& light, RGB& ambientLight, ReflectionMethod reflMethod) {
    if (depth > MAX_DEPTH) return RGB(0, 0, 0); // Prevent infinite recursion

    // Find the closest object the ray hits
    Shape* closestObject = nullptr;
    double minT = std::numeric_limits<double>::max();
    Material* hitMaterial = nullptr;
    Vec3 hitPoint;

    for (Shape* obj : objects) {
        double t = obj->solveRoot(ray);
        if (t > 0 && t < minT) {
            minT = t;
            closestObject = obj;
            hitMaterial = closestObject->getMaterial();
            hitPoint = ray.origin + (ray.direction * minT);
        }
    } 

    if (!closestObject) return RGB(0, 0, 0); // No intersection, return background

    if (hitMaterial) {
        if (dynamic_cast<CheckerboardMaterial*>(hitMaterial)) {
            // Apply checkerboard pattern
            CheckerboardMaterial* checkerboard = dynamic_cast<CheckerboardMaterial*>(hitMaterial);
            return checkerboard->getColorAtPoint(hitPoint);
        }
        
        if (dynamic_cast<SoccerBallMaterial*>(hitMaterial)) {
            // Handle soccer ball material with improved lighting
            SoccerBallSphere* soccerSphere = dynamic_cast<SoccerBallSphere*>(closestObject);
            if (soccerSphere) {
                RGB patternColor = soccerSphere->getColorAtPoint(hitPoint);
                
                // Apply enhanced Phong lighting to the pattern color
                Vec3 normal = closestObject->getSurfaceNormal(hitPoint);
                normal.normalize();
                
                Vec3 L = (light.position - hitPoint);
                L.normalize();
                Vec3 V = (ray.origin - hitPoint);
                V.normalize();
                Vec3 R = (normal * (2 * normal.dot(L))) - L;
                R.normalize();
                
                double mA = hitMaterial->getAmbientCoefficient();
                double mD = hitMaterial->getDiffusionCoefficient();
                double mS = hitMaterial->getSpecularCoefficient();
                double mSp = hitMaterial->getShininessCoefficient();
                
                // Enhanced lighting calculation
                RGB ambient = patternColor * ambientLight * mA;
                RGB diffuse = patternColor * light.intensity * mD * std::max(L.dot(normal), 0.0);
                
                // Add subtle subsurface scattering effect for more realistic soccer ball
                double subsurface = std::max(0.0, -L.dot(normal)) * 0.3;
                RGB subsurfaceColor = patternColor * light.intensity * subsurface;
                
                // Enhanced specular with color-dependent highlights
                RGB specularHighlightColor = RGB(230, 230, 230); // Brighter, more consistent white highlight
                RGB specular = specularHighlightColor * light.intensity * mS * pow(std::max(V.dot(R), 0.0), mSp);
                
                RGB finalColor = ambient + diffuse + subsurfaceColor + specular;
                
                // Shadow check with soft shadow effect
                Vec3 shadowOrigin = hitPoint + normal * EPS;
                Vec3 shadowDir = (light.position - shadowOrigin);
                double lightDistance = shadowDir.norm();
                shadowDir.normalize();
                Ray shadowRay(shadowOrigin, shadowDir);
                
                bool inShadow = false;
                for (Shape* obj : objects) {
                    if (obj != closestObject) {
                        double t_shadow = obj->solveRoot(shadowRay);
                        if (t_shadow > EPS && t_shadow < lightDistance - EPS) {
                            inShadow = true;
                            break;
                        }
                    }
                }
                
                if (inShadow) {
                    // Make shadows a bit more pronounced but retain some colored ambient and subsurface
                    finalColor = ambient * 0.9 + subsurfaceColor * 0.6; // Adjusted shadow color
                }
                
                finalColor.clamp();
                return finalColor;
            }
        }
    }

    // Compute intersection point and normal
    Vec3 normal = closestObject->getSurfaceNormal(hitPoint);
    normal.normalize();

    // Get material properties
    RGB objectColor = hitMaterial->getObjectColor();
    double mA = hitMaterial->getAmbientCoefficient();
    double mD = hitMaterial->getDiffusionCoefficient();
    double mS = hitMaterial->getSpecularCoefficient();
    double mSp = hitMaterial->getShininessCoefficient();

    // Light vector
    Vec3 L = (light.position - hitPoint);
    L.normalize();

    // View vector
    Vec3 V = (ray.origin - hitPoint);
    V.normalize();

    // Reflection vector
    Vec3 R = (normal * (2 * normal.dot(L))) - L;
    R.normalize();

    // **Phong Lighting Model**
    RGB ambient = objectColor * ambientLight * mA;
    RGB diffuse = objectColor * light.intensity * mD * std::max(L.dot(normal), 0.0);
    RGB specular = RGB(1, 1, 1) * light.intensity * mS * pow(std::max(V.dot(R), 0.0), mSp);

    RGB phongColor = ambient + diffuse + specular;

    // **Shadow Check**
    Vec3 shadowOrigin = hitPoint + normal * EPS; // Renamed for clarity
    Vec3 shadowDir = (light.position - shadowOrigin);
    shadowDir.normalize();
    Ray shadowRay(shadowOrigin, shadowDir);

    double t_shadow = closestObject->solveRoot(shadowRay); // Renamed t to t_shadow
    if (t_shadow > EPS && t_shadow < (light.position - shadowOrigin).norm()) {  
        phongColor = ambient;  // Only ambient light if blocked
    }

    // **Reflection & Refraction**
    double n1 = 1.0;   // Air
    double n2 = hitMaterial->getRefractionIndex();  // Glass, water, etc.

    RGB reflectionColor(0, 0, 0), refractionColor(0, 0, 0);

    if (n2 < 10) {
        bool inside = (ray.direction.dot(normal) > 0);
        if (inside) {
            std::swap(n1, n2);
            normal = -normal;
        }

        double eta = n1 / n2;
        double cosTheta1 = -normal.dot(ray.direction);
        // Removed redundant normal flip and cosTheta1 adjustment here as it's handled by the first 'inside' block

        double sin2Theta2 = eta * eta * (1.0 - cosTheta1 * cosTheta1);
        
        // **Total Internal Reflection (TIR)**
        if (sin2Theta2 > 1.0) {
            // Total Internal Reflection
            Vec3 reflectionDir = ray.direction - normal * (2.0 * ray.direction.dot(normal));
            reflectionDir.normalize();
            Ray reflectedRay(hitPoint + normal * EPS, reflectionDir);
            reflectionColor = traceRay(reflectedRay, screen, objects, depth + 1, light, ambientLight, reflMethod);
        } 
        else {
            // Refraction Calculation (Snell's Law)
            double cosTheta2 = sqrt(fabs(1.0 - sin2Theta2)); // Added fabs for robustness
            Vec3 refractionDir = (ray.direction * eta) + normal * (eta * cosTheta1 - cosTheta2);
            refractionDir.normalize();
            Ray refractedRay(hitPoint - normal * EPS, refractionDir); // Use -normal for ray exiting
            refractionColor = traceRay(refractedRay, screen, objects, depth + 1, light, ambientLight, reflMethod);
        
            // Reflection Calculation
            Vec3 reflectionDir = ray.direction - normal * (2.0 * ray.direction.dot(normal));
            reflectionDir.normalize();
            Ray reflectedRay(hitPoint + normal * EPS, reflectionDir); // Use +normal for ray reflecting
            reflectionColor = traceRay(reflectedRay, screen, objects, depth + 1, light, ambientLight, reflMethod);
        
            // Fresnel Reflectance
            double R = 1.0;
            double T = 0.0;
            if (reflMethod == FRESNEL) {
                double RsBase = (n1 * cosTheta1 - n2 * cosTheta2) / (n1 * cosTheta1 + n2 * cosTheta2);
                double Rs = RsBase * RsBase;
                double RpBase = (n1 * cosTheta2 - n2 * cosTheta1) / (n1 * cosTheta2 + n2 * cosTheta1);
                double Rp = RpBase * RpBase;
                R = Rs + Rp;
                T = 1.0 - R;
            }
            // Fresnel Reflectance approximated by Schlick's Method
            else if (reflMethod == SCHLICK) {
                double R0 = (n1 - n2) / (n1 + n2);
                R0 = R0 * R0;
                R = R0 + (1.0 - R0) * pow(1.0 - cosTheta1, 5);
                T = 1.0 - R;
            }
        
            RGB finalColor = phongColor + (reflectionColor * R) + (refractionColor * T);
            finalColor.clamp();
            return finalColor;
        }
    }

    RGB tmpColor = phongColor + reflectionColor;
    tmpColor.clamp();
    return tmpColor;
}

void processScreen(Screen& screen, Vec3& origin, vector<Shape*>& objects, LightSource& light, RGB& ambientLight, ANTI_ALIASING aa, ReflectionMethod reflMethod) {
    #pragma omp parallel for
    for (auto& pixel : screen) {
        double aspectRatio = (double)screen.width / screen.height;
        stack<Ray> st;

        auto computePixelPosition = [&](double px, double py) -> Vec3 {
            Vec3 screenCenter = origin + screen.normal * screen.pov;
            Vec3 pixelPosition = screenCenter 
                + screen.right * ((px - screen.width / 2.0) / screen.width * aspectRatio) 
                + screen.up * ((screen.height / 2.0 - py) / screen.height); //Inverted y direction.
            return pixelPosition;
        };
        
        auto shootRay = [&](double px, double py) {
            Vec3 pixelPos = computePixelPosition(px, py);
            Ray ray(origin, pixelPos - origin);
            ray.direction.normalize();
            st.push(ray);
        };

        switch (aa) {
            case AA_NONE: {
                shootRay(pixel.x, pixel.y);
                break;
            }
        
            case AA_RANDOM_2: {
                for (int i = 0; i < 2; i++) {
                    double randX = pixel.x + (rand() / (double)RAND_MAX - 0.5);
                    double randY = pixel.y + (rand() / (double)RAND_MAX - 0.5);
                    shootRay(randX, randY);
                }
                break;
            }
        
            case AA_REGULAR_2: {
                for (int i = 0; i < 2; i++) {
                    double offsetX = (i * 0.5) - 0.25;
                    shootRay(pixel.x + offsetX, pixel.y);
                }
                break;
            }
        
            case AA_RANDOM_4: {
                for (int i = 0; i < 4; i++) {
                    double randX = pixel.x + (rand() / (double)RAND_MAX - 0.5);
                    double randY = pixel.y + (rand() / (double)RAND_MAX - 0.5);
                    shootRay(randX, randY);
                }
                break;
            }
        
            case AA_REGULAR_4: {
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        double offsetX = (i * 0.5) - 0.25;
                        double offsetY = (j * 0.5) - 0.25;
                        shootRay(pixel.x + offsetX, pixel.y + offsetY);
                    }
                }
                break;
            }
        }

        // Loop through normal rays
        int rayCount = st.size();
        RGB accumulatedColor(0, 0, 0);

        // Process each ray
        while (!st.empty()) {
            Ray ray = st.top();
            st.pop();

            RGB color = traceRay(ray, screen, objects, 0, light, ambientLight, reflMethod);
            accumulatedColor += color;
        }

        // Average the accumulated color over the samples
        pixel.color = accumulatedColor / rayCount;
        pixel.color.clamp(); // Ensure RGB values stay within valid range
    }
}  

vector<Shape*> loadOBJ(const string& filename, Material* material) {
    vector<Vec3> vertices;
    vector<Vec3> vertexNormals; // Accumulated vertex normals
    vector<tuple<int, int, int>> faces;
    vector<Shape*> triangles;

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open OBJ file: " << filename << endl;
        return triangles;
    }

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        string prefix;
        iss >> prefix;

        if (prefix == "v") {
            double x, y, z;
            iss >> x >> y >> z;
            vertices.emplace_back(x, y, z);
            vertexNormals.emplace_back(0, 0, 0); // Initialize with zero vector
        } else if (prefix == "f") {
            int v1, v2, v3;
            iss >> v1 >> v2 >> v3;
            faces.emplace_back(v1 - 1, v2 - 1, v3 - 1); // Store as 0-based index
        }
    }

    // Accumulate normals
    for (const auto& [i1, i2, i3] : faces) {
        Vec3& p1 = vertices[i1];
        Vec3& p2 = vertices[i2];
        Vec3& p3 = vertices[i3];

        Vec3 faceNormal = (p2 - p1).cross(p3 - p1);

        vertexNormals[i1] = vertexNormals[i1] + faceNormal;
        vertexNormals[i2] = vertexNormals[i2] + faceNormal;
        vertexNormals[i3] = vertexNormals[i3] + faceNormal;
    }

    // Normalize vertex normals
    for (auto& n : vertexNormals) {
        n.normalize();
    }

    // Create triangles with vertex normals
    for (const auto& [i1, i2, i3] : faces) {
        Triangle* tri = new Triangle(vertices[i1], vertices[i2], vertices[i3], material);
        tri->n1 = vertexNormals[i1];
        tri->n2 = vertexNormals[i2];
        tri->n3 = vertexNormals[i3];
        tri->normal = (tri->v2 - tri->v1).cross(tri->v3 - tri->v1);
        tri->normal.normalize();
        triangles.push_back(tri);
    }

    file.close();
    return triangles;
}

// New function to load waypoints and objects (wall) from CSV
void loadWaypointsAndObjectsFromCSV(
    const std::string& filename,
    std::vector<Vec3>& ballWaypoints_simCoords, // Output: ball trajectory points (simulation coordinates)
    std::vector<Vec3>& ballSpins_simCoords,     // Output: ball spin vectors (simulation coordinates)
    std::vector<Shape*>& objects,               // Input/Output: scene objects, to add wall to
    Material* wallMaterial,                     // Input: material for the wall
    Vec3& goalPosition_simCoords                // Output: goal position (simulation coordinates)
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << filename << std::endl;
        return;
    }

    std::string line;
    // Skip header
    if (!std::getline(file, line)) {
        std::cerr << "CSV file is empty or header is missing: " << filename << std::endl;
        return;
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> segments;
        while(std::getline(ss, segment, ',')) {
           segments.push_back(segment);
        }

        if (segments.size() < 9) { // Expecting at least 9 columns for ball or wall
            // std::cerr << "Skipping malformed CSV line (not enough segments): " << line << std::endl;
            continue;
        }

        try {
            std::string object_type = segments[3];
            double csv_x = std::stod(segments[0]);
            double csv_y = std::stod(segments[1]);
            double csv_z = std::stod(segments[2]);

            if (object_type == "ball") {
                ballWaypoints_simCoords.emplace_back(csv_x, csv_y, csv_z); // Store in simulation coordinates
                // Assuming orientation_x,y,z columns (6,7,8) are used for spin for ball objects
                double spin_x = std::stod(segments[6]);
                double spin_y = std::stod(segments[7]);
                double spin_z = std::stod(segments[8]);
                ballSpins_simCoords.emplace_back(spin_x, spin_y, spin_z);
            } else if (object_type == "goal") {
                goalPosition_simCoords.x = csv_x;
                goalPosition_simCoords.y = csv_y;
                goalPosition_simCoords.z = csv_z;
            } else if (object_type == "wall") {
                // Wall properties from CSV (simulation coordinates)
                // segments[0-2] are csv_x, csv_y, csv_z (center_sim)
                // segments[4] is size_x (width_sim)
                // segments[5] is size_y (height_sim)
                // segments[6-8] are orientation_x,y,z (normal_sim)

                double width_sim = std::stod(segments[4]);
                double height_sim = std::stod(segments[5]);
                Vec3 normal_sim(std::stod(segments[6]), std::stod(segments[7]), std::stod(segments[8]));

                // Convert center and normal to world coordinates
                Vec3 center_world;
                center_world.x = csv_x;       // depth
                center_world.y = csv_z;       // width
                center_world.z = csv_y;       // height

                Vec3 normal_world;
                normal_world.x = normal_sim.x; // depth component
                normal_world.y = normal_sim.z; // width component
                normal_world.z = normal_sim.y; // height component
                normal_world.normalize();

                // Dimensions in world (assuming direct mapping for width/height of the panel)
                double wall_width_world = width_sim;
                double wall_height_world = height_sim;

                // Create orthonormal basis for the wall plane
                Vec3 u_vec; // "right" vector on the wall plane
                Vec3 v_vec; // "up" vector on the wall plane
                Vec3 w_vec = normal_world; // normal vector

                // Choose a vector not parallel to w_vec to start cross products
                Vec3 temp_aux_vec = (std::abs(w_vec.x) > 0.1 ? Vec3(0,1,0) : Vec3(1,0,0));
                if (std::abs(w_vec.dot(temp_aux_vec)) > 0.99) { // If still too parallel (e.g. w_vec is Y axis, temp_aux_vec is Y axis)
                    temp_aux_vec = Vec3(0,0,1); // Try Z-axis
                     if (std::abs(w_vec.dot(temp_aux_vec)) > 0.99) { // If still too parallel (e.g. w_vec is Z axis)
                         temp_aux_vec = Vec3(1,0,0); // Should be different now
                     }
                }


                u_vec = temp_aux_vec.cross(w_vec);
                if (u_vec.norm() < EPS) { // Handles case where w_vec and temp_aux_vec are parallel
                    // Fallback: if normal is, e.g., (0,1,0), temp_aux_vec might be (0,1,0)
                    // Try a different auxiliary vector
                    if (std::abs(w_vec.y) > 0.9) temp_aux_vec = Vec3(1,0,0); // If normal is Y-ish, use X
                    else temp_aux_vec = Vec3(0,1,0); // Otherwise use Y
                    u_vec = temp_aux_vec.cross(w_vec);
                }
                u_vec.normalize();
                v_vec = w_vec.cross(u_vec);
                v_vec.normalize(); // v_vec is already normalized if w_vec and u_vec are ortho-normalized

                // Calculate 4 corner vertices of the wall panel
                Vec3 v0 = center_world - u_vec * (wall_width_world / 2.0) - v_vec * (wall_height_world / 2.0);
                Vec3 v1 = center_world + u_vec * (wall_width_world / 2.0) - v_vec * (wall_height_world / 2.0);
                Vec3 v2 = center_world + u_vec * (wall_width_world / 2.0) + v_vec * (wall_height_world / 2.0);
                Vec3 v3 = center_world - u_vec * (wall_width_world / 2.0) + v_vec * (wall_height_world / 2.0);

                // Create two triangles for the wall panel
                Triangle* tri1 = new Triangle(v0, v1, v2, wallMaterial);
                Triangle* tri2 = new Triangle(v0, v2, v3, wallMaterial);
                
                objects.push_back(tri1);
                objects.push_back(tri2);
            }
        } catch (const std::invalid_argument& ia) {
            // std::cerr << "Invalid argument: " << ia.what() << " on line: " << line << std::endl;
        } catch (const std::out_of_range& oor) {
            // std::cerr << "Out of range: " << oor.what() << " on line: " << line << std::endl;
        }
    }
    file.close();
}

void renderMovingSphere(const std::vector<Vec3>& ballWaypoints_simCoords, 
                        const std::vector<Vec3>& ballSpins_simCoords,
                        Vec3 camPos, 
                        std::vector<Shape*>& objects, 
                        SoccerBallMaterial* ballMaterial,  // Back to SoccerBallMaterial*
                        LightSource& light, 
                        RGB ambientLight, 
                        ANTI_ALIASING aa, 
                        ReflectionMethod reflMethod,
                        const std::string& outputDir) { // Added outputDir parameter
    
    // Create soccer ball sphere with the provided material
    SoccerBallSphere* movingSphere = new SoccerBallSphere(Vec3(0,0,0), 0.11, ballMaterial);
    objects.push_back(movingSphere);
    
    int numFrames = ballWaypoints_simCoords.size();
    if (numFrames == 0) {
        std::cerr << "No waypoints provided for renderMovingSphere." << std::endl;
        objects.erase(std::remove(objects.begin(), objects.end(), movingSphere), objects.end());
        delete movingSphere;
        return;
    }
    if (ballSpins_simCoords.size() != numFrames) {
        std::cerr << "Mismatch between waypoint count and spin data count." << std::endl;
    }

    for (int i = 0; i < numFrames; ++i) {
        const Vec3& waypoint_sim = ballWaypoints_simCoords[i];
        // Apply coordinate mapping: CSV (depth, height, width) to world (depth, width, height)
        movingSphere->center.x = waypoint_sim.x; // X (depth) maps directly
        movingSphere->center.y = waypoint_sim.z; // Y (world width) gets Z from CSV waypoint (width)
        movingSphere->center.z = waypoint_sim.y; // Z (world height) gets Y from CSV waypoint (height)

        // Account for spin
        if (i < ballSpins_simCoords.size()) {
            const Vec3& spin_sim = ballSpins_simCoords[i];
            Vec3 spin_world;
            spin_world.x = spin_sim.x; // X (depth) maps directly
            spin_world.y = spin_sim.z; // Y (world width) gets Z from CSV spin (width component)
            spin_world.z = spin_sim.y; // Z (world height) gets Y from CSV spin (height component)

            double rot_angle = spin_world.norm();
            Vec3 rot_axis_world = spin_world;

            if (rot_angle > EPS) {
                rot_axis_world = rot_axis_world / rot_angle; // Normalize the axis
            } else {
                rot_axis_world = Vec3(0, 0, 1); // Default axis if no spin
                rot_angle = 0.0;
            }
            movingSphere->currentRotationAxis = rot_axis_world;
            movingSphere->currentRotationAngle = rot_angle; 
        }
        
        // Calculate screen normal to look at the ball
        Vec3 screenNormal = movingSphere->center - camPos;
        if (screenNormal.norm() < EPS) {
            screenNormal = Vec3(1,0,0); // Default to looking along X-axis
        } else {
            screenNormal.normalize();
        }
        
        Screen screen(screenNormal, 2560, 1440);
        screen.pov = 0.9; 
        processScreen(screen, camPos, objects, light, ambientLight, aa, reflMethod);
        std::ostringstream filenameStream;
        filenameStream << outputDir << "/frame_" << std::setw(4) << std::setfill('0') << i << ".jpg"; // Use outputDir
        screen.writeToJPG(filenameStream.str());
    }
    objects.erase(std::remove(objects.begin(), objects.end(), movingSphere), objects.end());
    delete movingSphere;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Not enough parameters. Usage: <program> <ReflectionMethod> <MAX_DEPTH>" << endl;
        exit(-1);
    }
    ReflectionMethod reflMethod = (ReflectionMethod)atoi(argv[1]);
    MAX_DEPTH = (int)atoi(argv[2]);

    std::string baseInputDir = "../01_KickSimulator/output/";
    std::string baseOutputDir = "./output/";

    // Create base output directory if it doesn't exist
    if (!std::filesystem::exists(baseOutputDir)) {
        std::filesystem::create_directories(baseOutputDir);
    }
    
    Vec3 origin(0.0, 0.0, 0.0); // Camera for first view (if used)
    double radius = 3.0;  
    double glassRadius = radius;  
    double smallSphereRadius = 0.25 * 2;  
    Vec3 glassCenter(10, 0, 0);  // Large sphere's position (Y stays constant)

    // Materials
    Glassy glassy(RGB(255, 255, 255));
    Metallic redMetal(RGB(255, 0, 0));
    Metallic blueMetal(RGB(0, 0, 255));
    Metallic greenMetal(RGB(0, 255, 0));
    Metallic yellowMetal(RGB(255, 255, 0));
    Metallic whiteMetal(RGB(255, 255, 255));
    CheckerboardMaterial checkerboard(RGB(34, 139, 34), RGB(0, 100, 0), radius); // Grassy green
    
    // Create enhanced soccer ball material with traditional black and white colors
    SoccerBallMaterial soccerBallMat(RGB(20, 20, 20), RGB(240, 240, 240)); // Slightly off-pure for more realism

    std::vector<Shape*> objects; 

    // **Ground Plane**
    Plane floor(Vec3(0, 0, -1), Vec3(0, 0, 1), &checkerboard);
    // objects.push_back(&floor); // Will be added after clearing objects

    // **Enhanced Light source with better positioning and intensity**
    LightSource light(Vec3(8, 8, 12), RGB(180, 180, 180)); // Increased intensity and better positioning
    RGB ambientLight(45, 45, 45); // Increased ambient light for better visibility

    for (const auto& entry : std::filesystem::directory_iterator(baseInputDir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.rfind("wavepoints_", 0) == 0 && filename.find(".csv") != std::string::npos) {
                std::cout << "Processing file: " << filename << std::endl;
                std::string currentWavepointFile = entry.path().string();
                std::string runName = filename.substr(0, filename.find(".csv"));
                std::string currentOutputDir = baseOutputDir + runName + "/";

                if (!std::filesystem::exists(currentOutputDir)) {
                    std::filesystem::create_directories(currentOutputDir);
                }

                objects.clear(); 
                objects.push_back(&floor); 

                std::vector<Vec3> ballWaypoints_simCoords;
                std::vector<Vec3> ballSpins_simCoords; 
                std::vector<Shape*> csv_wall_triangles_managed; 

                size_t objects_count_before_wall = objects.size();
                Vec3 goalPosition_simCoords(0,0,0); 
                loadWaypointsAndObjectsFromCSV(currentWavepointFile, 
                                               ballWaypoints_simCoords, ballSpins_simCoords, objects, &whiteMetal, goalPosition_simCoords);
                size_t objects_count_after_wall = objects.size();

                for (size_t i = objects_count_before_wall; i < objects_count_after_wall; ++i) {
                    csv_wall_triangles_managed.push_back(objects[i]);
                }

                Vec3 goalkeeper_pos_world; 
                goalkeeper_pos_world.x = goalPosition_simCoords.x; 
                goalkeeper_pos_world.y = goalPosition_simCoords.z; 
                goalkeeper_pos_world.z = goalPosition_simCoords.y; 

                if (!csv_wall_triangles_managed.empty()) {
                    bool found_intersect_point = false;
                    for (size_t waypoint_idx = 0; waypoint_idx < ballWaypoints_simCoords.size(); ++waypoint_idx) {
                        const auto& waypoint_sim = ballWaypoints_simCoords[waypoint_idx];
                        Vec3 ball_world_pos;
                        ball_world_pos.x = waypoint_sim.x; 
                        ball_world_pos.y = waypoint_sim.z; 
                        ball_world_pos.z = waypoint_sim.y; 

                        Ray los_ray(goalkeeper_pos_world, ball_world_pos - goalkeeper_pos_world);
                        los_ray.direction.normalize();
                        double dist_to_ball = (ball_world_pos - goalkeeper_pos_world).norm();
                        bool obstructed = false;

                        for (Shape* wall_shape : csv_wall_triangles_managed) {
                            double t_intersect = wall_shape->solveRoot(los_ray);
                            if (t_intersect > EPS && t_intersect < (dist_to_ball - EPS)) {
                                obstructed = true;
                                break;
                            }
                        }

                        if (!obstructed) {
                            std::string wallIntersectFile = currentOutputDir + "wall_intersect.csv";
                            std::ofstream outfile(wallIntersectFile);
                            if (outfile.is_open()) {
                                outfile << "sim_x,sim_y,sim_z,waypoint_index" << std::endl; 
                                outfile << std::fixed << std::setprecision(6)
                                        << waypoint_sim.x << ","
                                        << waypoint_sim.y << ","
                                        << waypoint_sim.z << ","
                                        << waypoint_idx << std::endl;
                                outfile.close();
                                found_intersect_point = true;
                            } else {
                                std::cerr << "Error: Could not open " << wallIntersectFile << " for writing." << std::endl;
                            }
                            break; 
                        }
                    }
                    if (!found_intersect_point) {
                        // ¯\\_(ツ)_/¯ 
                    }
                }
    
                renderMovingSphere(ballWaypoints_simCoords, ballSpins_simCoords, goalkeeper_pos_world, objects, &soccerBallMat, light, ambientLight, AA_REGULAR_4, reflMethod, currentOutputDir);

                for (Shape* wall_tri : csv_wall_triangles_managed) {
                    objects.erase(std::remove(objects.begin(), objects.end(), wall_tri), objects.end());
                    delete wall_tri;
                }
                csv_wall_triangles_managed.clear();
                
                std::cout << "Finished processing for " << filename << ". Output in " << currentOutputDir << std::endl;
            }
        }
    }

    // Cleanup other dynamically allocated objects if any (e.g., teapot triangles if loaded)
    // for (Shape* shape : teapot_triangles) { delete shape; }
    // teapot_triangles.clear();

    return 0;
}