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

    Sphere(Vec3 _center, double _R, Material* _material) : center(_center), R(_R), material(_material) {}

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

        // Use Newton’s method to solve
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

        pov = normal.norm();

        // Initialize pixels with default values
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                pixels.emplace_back(x, y, RGB(0, 0, 0));
            }
        }

        // Step 1: Define Simple Vectors (Before Rotation)
        Vec3 simple_normal(1, 0, 0);
        Vec3 simple_right(0, 1, 0);
        Vec3 simple_up(0, 0, 1);

        // Step 2: Compute Rotation Axis and Angle
        Vec3 rotation_axis = simple_normal.cross(normal);
        double rotation_angle = acos(simple_normal.dot(normal) / (simple_normal.norm() * normal.norm()));

        // Step 3: Rotate Vectors
        if (rotation_axis.norm() > EPS) { // Check if rotation is needed
            right = rotate(simple_right, rotation_axis, rotation_angle);
            up = rotate(simple_up, rotation_axis, rotation_angle);
            this->normal = rotate(simple_normal, rotation_axis, rotation_angle); // Update screen normal
        } else { // No rotation needed or normal is parallel
            right = simple_right;
            up = simple_up;
            this->normal = simple_normal;
             if (simple_normal.dot(normal) < 0) { // Flipped normal
                right = -right; // Adjust orientation if normal was flipped
                this->normal = -this->normal;
            }
        }
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
    Vec3 shadowOrigin = hitPoint + normal * EPS; // Prevent self-intersection
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
    Material* wallMaterial                      // Input: material for the wall
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

        if (segments.size() < 12) { // Expecting at least 12 columns
            // std::cerr << "Skipping malformed CSV line: " << line << std::endl;
            continue;
        }

        try {
            std::string object_type = segments[3];
            double csv_x = std::stod(segments[0]);
            double csv_y = std::stod(segments[1]);
            double csv_z = std::stod(segments[2]);

            if (object_type == "ball") {
                ballWaypoints_simCoords.emplace_back(csv_x, csv_y, csv_z); // Store in simulation coordinates
                double spin_x = std::stod(segments[9]);
                double spin_y = std::stod(segments[10]);
                double spin_z = std::stod(segments[11]);
                ballSpins_simCoords.emplace_back(spin_x, spin_y, spin_z);
            } else if (object_type == "wall") {
                double wall_width = std::stod(segments[4]);
                double wall_height = std::stod(segments[5]);
                double orientation_csv_x = std::stod(segments[6]);
                double orientation_csv_y = std::stod(segments[7]);
                double orientation_csv_z = std::stod(segments[8]);

                // Convert CSV wall coordinates and orientation to world coordinates
                // CSV: x (depth), y (height), z (width)
                // World: x (depth), y (width), z (height)
                Vec3 center_world(csv_x, csv_z, csv_y); 
                Vec3 normal_sim(orientation_csv_x, orientation_csv_y, orientation_csv_z);
                Vec3 normal_world(normal_sim.x, normal_sim.z, normal_sim.y);
                normal_world.normalize();

                // Determine wall's local coordinate system in world space
                Vec3 wall_surface_up_world;
                Vec3 wall_surface_right_world;

                // Assuming wall is mostly upright, its "up" is along world Z.
                // If normal is (anti)parallel to world Z, wall is horizontal, use world Y as "up".
                if (std::abs(normal_world.dot(Vec3(0,0,1))) > 0.99) { // Wall is horizontal or nearly so
                    wall_surface_up_world = Vec3(0,1,0); // Use world Y as surface "up"
                    wall_surface_right_world = wall_surface_up_world.cross(normal_world);
                    if (wall_surface_right_world.norm() < EPS) { // normal_world is parallel to world Y
                        wall_surface_right_world = Vec3(1,0,0).cross(normal_world); // Use world X instead
                    }
                } else { // Wall is upright or tilted
                    wall_surface_up_world = Vec3(0,0,1); // Use world Z as surface "up"
                    wall_surface_right_world = wall_surface_up_world.cross(normal_world);
                }
                wall_surface_right_world.normalize();
                // Recalculate up to ensure orthogonality, normal_world is "outward" normal
                wall_surface_up_world = normal_world.cross(wall_surface_right_world);
                wall_surface_up_world.normalize();


                Vec3 v_bl = center_world - wall_surface_right_world * (wall_width / 2.0) - wall_surface_up_world * (wall_height / 2.0);
                Vec3 v_br = center_world + wall_surface_right_world * (wall_width / 2.0) - wall_surface_up_world * (wall_height / 2.0);
                Vec3 v_tr = center_world + wall_surface_right_world * (wall_width / 2.0) + wall_surface_up_world * (wall_height / 2.0);
                Vec3 v_tl = center_world - wall_surface_right_world * (wall_width / 2.0) + wall_surface_up_world * (wall_height / 2.0);

                Triangle* wall_tri1 = new Triangle(v_bl, v_br, v_tr, wallMaterial);
                Triangle* wall_tri2 = new Triangle(v_bl, v_tr, v_tl, wallMaterial);
                
                // Set normals for smooth shading (though wall is flat)
                wall_tri1->normal = normal_world; wall_tri1->n1 = normal_world; wall_tri1->n2 = normal_world; wall_tri1->n3 = normal_world;
                wall_tri2->normal = normal_world; wall_tri2->n1 = normal_world; wall_tri2->n2 = normal_world; wall_tri2->n3 = normal_world;

                objects.push_back(wall_tri1);
                objects.push_back(wall_tri2);
            }
        } catch (const std::invalid_argument& ia) {
            // std::cerr << "Invalid argument: " << ia.what() << " on line: " << line << std::endl;
        } catch (const std::out_of_range& oor) {
            // std::cerr << "Out of range: " << oor.what() << " on line: " << line << std::endl;
        }
    }
    file.close();
}


void renderMovingSphere(const std::vector<Vec3>& ballWaypoints_simCoords, Vec3 camPos, std::vector<Shape*>& objects, Material* sphereMat, LightSource& light, RGB ambientLight, ANTI_ALIASING aa, ReflectionMethod reflMethod) {
    Sphere* movingSphere = new Sphere(Vec3(0,0,0), 0.11, sphereMat);
    objects.push_back(movingSphere);
    
    int numFrames = ballWaypoints_simCoords.size();
    if (numFrames == 0) {
        std::cerr << "No waypoints provided for renderMovingSphere." << std::endl;
        // Clean up sphere if added
        objects.erase(std::remove(objects.begin(), objects.end(), movingSphere), objects.end());
        delete movingSphere;
        return;
    }

    for (int i = 0; i < numFrames; ++i) {
        const Vec3& waypoint_sim = ballWaypoints_simCoords[i];
        // Apply coordinate mapping: CSV (depth, height, width) to world (depth, width, height)
        movingSphere->center.x = waypoint_sim.x; // X (depth) maps directly
        movingSphere->center.y = waypoint_sim.z; // Y (world width) gets Z from CSV waypoint (width)
        movingSphere->center.z = waypoint_sim.y; // Z (world height) gets Y from CSV waypoint (height)
        
        // Dynamically calculate screenNormal to look at the ball
        Vec3 screenNormal = movingSphere->center - camPos;
        if (screenNormal.norm() < EPS) { // Avoid issues if camera is at the ball's position
            screenNormal = Vec3(1,0,0); // Default to looking along X-axis
        } else {
            screenNormal.normalize();
        }
        
        Screen screen(screenNormal, 2560, 1440);
        screen.pov = 0.9; 
        processScreen(screen, camPos, objects, light, ambientLight, aa, reflMethod);
        std::ostringstream filenameStream;
        filenameStream << "output/frame_" << std::setw(4) << std::setfill('0') << i << ".jpg";
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

    std::vector<Shape*> objects; 

    // Load Teapot (example, if obj file exists and is used)
    // auto teapot_triangles = loadOBJ("teapot.obj", &redMetal);
    // objects.insert(objects.end(), teapot_triangles.begin(), teapot_triangles.end());
    // cout << "Loaded teapot: " << teapot_triangles.size() << " triangles have been loaded"  << endl;
    // { // Transformations for teapot if loaded
    //     Vec3 rotationAxis(0, 1, 0); double angle = M_PI / 2;
    //     Vec3 teapotOffset(10, 0, -1); Vec3 rotationAxis2(1,0,0); double angle2 = M_PI/2;
    //     for (Shape* shape : teapot_triangles) { /* ... apply rotations and translations ... */ }
    // }


    // **Ground Plane**
    Plane floor(Vec3(0, 0, -1), Vec3(0, 0, 1), &checkerboard);
    // objects.push_back(&floor); // Will be added after clearing objects

    // **Light source**
    LightSource light(Vec3(5, 5, 10), RGB(255, 255, 255)); // Adjusted light height
    RGB ambientLight(50, 50, 50); // Adjusted ambient light

    objects.clear(); 
    objects.push_back(&floor); 

    std::vector<Vec3> ballWaypoints_simCoords;
    std::vector<Vec3> ballSpins_simCoords; // Currently unused for rendering
    std::vector<Shape*> csv_wall_triangles_managed; // To store pointers for cleanup

    // Load waypoints and wall from CSV
    // Store current object count to identify added wall triangles
    size_t objects_count_before_wall = objects.size();
    loadWaypointsAndObjectsFromCSV("/home/anthony/dev/Advanced-Simulation-in-Natural-Sciences/Free-kick-simulator/01_KickSimulator/output/wavepoints_0.csv", 
                                   ballWaypoints_simCoords, ballSpins_simCoords, objects, &whiteMetal);
    size_t objects_count_after_wall = objects.size();

    for (size_t i = objects_count_before_wall; i < objects_count_after_wall; ++i) {
        csv_wall_triangles_managed.push_back(objects[i]);
    }
    
    // Define camera positions and screen normals (as Vec3)
    // Vec3 freekick_cam_pos = origin; // Example: camera at origin for free kick view
    // Vec3 freekick_screen_normal(1.0, 0.0, 0.0);

    Vec3 goalkeeper_pos(0.0, 0.0, 0.5); // Goalkeeper camera slightly above ground
    // Vec3 goalkeeper_screen_normal(1.0, 0.0, 0.0); // Looking along +X (depth) - No longer needed here

    // Render from goalkeeper's perspective using CSV waypoints
    // The number of frames is determined by the number of waypoints in the CSV.
    renderMovingSphere(ballWaypoints_simCoords, goalkeeper_pos, objects, &whiteMetal, light, ambientLight, AA_REGULAR_4, reflMethod);

    // Cleanup dynamically allocated wall triangles from CSV
    for (Shape* wall_tri : csv_wall_triangles_managed) {
        objects.erase(std::remove(objects.begin(), objects.end(), wall_tri), objects.end());
        delete wall_tri;
    }
    csv_wall_triangles_managed.clear();

    // Cleanup other dynamically allocated objects if any (e.g., teapot triangles if loaded)
    // for (Shape* shape : teapot_triangles) { delete shape; }
    // teapot_triangles.clear();


    return 0;
}