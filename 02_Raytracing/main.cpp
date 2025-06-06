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
#include <sstream> 
#include <string>  
#include <algorithm>
#include <limits> 
#include <filesystem> 
#include <map> 
#include "../01_KickSimulator/Vector.h" 

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

// Represents an RGB color with integer components (0-255).
// Supports basic arithmetic operations and clamping.
class RGB {
public:
    int r, g, b;

    RGB() : r(0), g(0), b(0) {}
    RGB(int _r, int _g, int _b) : r(_r), g(_g), b(_b) {}

    RGB operator+(const RGB& other) const {
        return RGB(r + other.r, g + other.g, b + other.b);
    }

    RGB& operator+=(const RGB& other) {
        r += other.r; g += other.g; b += other.b;
        return *this;
    }

    RGB operator-(const RGB& other) const {
        return RGB(r - other.r, g - other.g, b - other.b);
    }

    RGB& operator-=(const RGB& other) {
        r -= other.r; g -= other.g; b -= other.b;
        return *this;
    }

    RGB operator*(const RGB& other) const {
        return RGB( static_cast<int>((static_cast<long long>(r) * other.r) / 255.0),
                    static_cast<int>((static_cast<long long>(g) * other.g) / 255.0),
                    static_cast<int>((static_cast<long long>(b) * other.b) / 255.0) );
    }

    RGB& operator*=(const RGB& other) {
        r = static_cast<int>((static_cast<long long>(r) * other.r) / 255.0);
        g = static_cast<int>((static_cast<long long>(g) * other.g) / 255.0);
        b = static_cast<int>((static_cast<long long>(b) * other.b) / 255.0);
        return *this;
    }

    RGB operator*(double scalar) const {
        return RGB(r * scalar, g * scalar, b * scalar);
    }

    RGB& operator*=(double scalar) {
        r *= scalar; g *= scalar; b *= scalar;
        return *this;
    }

    RGB operator/(double scalar) const {
        return (scalar != 0) ? RGB(r / scalar, g / scalar, b / scalar) : RGB();
    }

    RGB& operator/=(double scalar) {
        if (scalar != 0) {
            r /= scalar; g /= scalar; b /= scalar;
        }
        return *this;
    }

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

    friend std::ostream& operator<<(std::ostream& os, const RGB& color) {
        os << "RGB(" << color.r << ", " << color.g << ", " << color.b << ")";
        return os;
    }
};

// Defines a point light source with a position and intensity.
class LightSource {
public:
    Vec3 position;
    RGB intensity;

    LightSource(Vec3 _position, RGB _intensity) : position(_position), intensity(_intensity) {}
};

// Represents a ray with an origin point and a direction vector.
class Ray {
public:
    Vec3 origin;
    Vec3 direction;
    Ray(Vec3 _o, Vec3 _d) : origin(_o), direction(_d) {}
};

// Implements Rodrigues' rotation formula to rotate a point around an axis.
Vec3 rotate(Vec3 p, Vec3 axis, double angle) {
    axis.normalize();
    return p * cos(angle) + axis.cross(p) * sin(angle) + 
           axis * (axis.dot(p)) * (1 - cos(angle));
}

// Represents a single pixel on the screen with coordinates and a color.
class Pixel {
public:
    int x, y;
    RGB color;

    Pixel(int _x, int _y, RGB _color = RGB()) : x(_x), y(_y), color(_color) {}
};

// Abstract base class for materials, defining properties for lighting calculations.
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

// Abstract base class for geometric shapes in the scene.
// Requires methods for normal calculation and ray intersection.
class Shape {
public:
    virtual Vec3 getSurfaceNormal(Vec3& pointOnSurface) = 0;
    virtual double solveRoot(Ray ray) = 0;
    virtual Material* getMaterial() const = 0;
    virtual ~Shape() {}
};

// Represents a sphere shape with a center, radius, and material.
// Includes logic for ray-sphere intersection and surface normal calculation.
class Sphere : public Shape {
public:
    Vec3 center;
    double R;
    Material* material;
    Vec3 currentRotationAxis; 
    double currentRotationAngle; 

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
            return -std::numeric_limits<double>::min();  
        }

        double inv2a = 0.5 / a;  
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

// Defines a quartic function: a*t^4 + b*t^3 + c*t^2 + d*t + e.
double quarticFunction(double t, double a, double b, double c, double d, double e) {
    return a * t * t * t * t + b * t * t * t + c * t * t + d * t + e;
}

// Defines the derivative of the quartic function.
double quarticDerivative(double t, double a, double b, double c, double d) {
    return 4 * a * t * t * t + 3 * b * t * t + 2 * c * t + d;
}

// Uses Newton's method to find real roots of a quartic equation.
// This is primarily used for ray-torus intersection.
double newtonSolveQuartic(double a, double b, double c, double d, double e, double initialGuess, int maxIterations = 50, double tolerance = 1e-6) {
    double t = initialGuess;
    
    for (int i = 0; i < maxIterations; i++) {
        double f_t = quarticFunction(t, a, b, c, d, e);
        double f_prime_t = quarticDerivative(t, a, b, c, d);

        if (std::abs(f_prime_t) < tolerance) {
            break; 
        }

        double t_next = t - f_t / f_prime_t;
        
        if (std::abs(t_next - t) < tolerance) {
            return t_next; 
        }
        
        t = t_next;
    }

    return std::numeric_limits<double>::max(); 
}

// Represents a torus shape with a center, major radius (R), minor radius (r), and material.
// Implements ray-torus intersection using a quartic solver.
class Torus : public Shape {
public:
    Vec3 center;
    double R; 
    double r; 
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
        Vec3 o = ray.origin - center;
        Vec3 d = ray.direction;

        double dDotD = d.dot(d);
        double oDotD = o.dot(d);
        double oDotO = o.dot(o);
        double sumR = R * R + r * r;

        double a = dDotD * dDotD;
        double b = 4.0 * dDotD * oDotD;
        double c = 4.0 * oDotD * oDotD + 2.0 * dDotD * (oDotO - sumR) + 4.0 * R * R * d.z * d.z;
        double d_coeff = 4.0 * (oDotO - sumR) * oDotD + 8.0 * R * R * o.z * d.z;
        double e = (oDotO - sumR) * (oDotO - sumR) - 4.0 * R * R * (r * r - o.z * o.z);

        double initialGuess = 1.0;
        double root = newtonSolveQuartic(a, b, c, d_coeff, e, initialGuess);

        if (root == std::numeric_limits<double>::max() || root < 0) {
            return -std::numeric_limits<double>::min();
        }

        return root;
    }
};

// Represents an infinite plane defined by a point on the plane and a normal vector.
class Plane : public Shape {
public:
    Vec3 point;    
    Vec3 normal;   
    Material* material; 

    Plane(Vec3 _point, Vec3 _normal, Material* _material) : point(_point), normal(_normal), material(_material) {}

    Material* getMaterial() const override { return material; }

    Vec3 getSurfaceNormal(Vec3& pointOnSurface) override {
        return normal; 
    }

    double solveRoot(Ray ray) override {
        double denominator = 1/ray.direction.dot(normal);
        
        if (fabs(denominator) < EPS) {
            return -std::numeric_limits<double>::max();  
        }
        
        double t = (point - ray.origin).dot(normal) * denominator;
        
        if (t > EPS) {
            return t;
        }
        
        return -std::numeric_limits<double>::max();  
    }
};

// Represents a Klein bottle, a non-orientable surface, defined parametrically.
// Intersection is approximated iteratively.
class KleinBottle : public Shape {
public:
    Vec3 center;
    double R; 
    double r; 
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
    
            double denom = normal.dot(ray.direction);
            if (fabs(denom) < 1e-6) { 
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

// Represents a triangle defined by three vertices.
// Supports ray-triangle intersection (Möller–Trumbore algorithm) and interpolated normals.
class Triangle : public Shape {
public:
    Vec3 v1;
    Vec3 v2;
    Vec3 v3;
    Vec3 normal;   
    Vec3 n1;
    Vec3 n2;
    Vec3 n3;
    Material* material; 

    Triangle(Vec3 _v1, Vec3 _v2, Vec3 _v3, Material* _material) : v1(_v1), v2(_v2), v3(_v3), material(_material) {
        normal = (v2 - v1).cross(v3 - v1);
        normal.normalize();
    }

    Material* getMaterial() const override { return material; }

    Vec3 getSurfaceNormal(Vec3& pointOnSurface) override {
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

        if (alpha < 0 || beta < 0 || gamma < 0 || std::isnan(alpha) || std::isnan(beta) || std::isnan(gamma)) { 
            return normal;  
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

// Represents the camera's view plane (screen).
// Manages pixel data, screen dimensions, orientation, and JPG output.
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

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                pixels.emplace_back(x, y, RGB(0, 0, 0));
            }
        }

        if (this->pov > EPS) {
            this->normal = this->normal / this->pov; 
        } else {
            this->normal = Vec3(1,0,0); 
            if (this->pov < EPS) this->pov = 1.0; 
        }

        Vec3 globalUpDirection(0, 0, 1); 

        this->right = globalUpDirection.cross(this->normal);
        if (this->right.norm() < EPS) {
            this->right = Vec3(1, 0, 0);
            if (this->normal.z > 0) { 
                 this->up = Vec3(0, -1, 0); 
            } else { 
                 this->up = Vec3(0, 1, 0); 
            }
            this->up.normalize(); 
            return; 

        } else {
            this->right.normalize();
        }

        this->up = this->normal.cross(this->right);
        this->up.normalize(); 
    }

    std::vector<Pixel>::iterator begin() { return pixels.begin(); }
    std::vector<Pixel>::iterator end() { return pixels.end(); }

    void writeToJPG(const std::string& filename, int quality = 90) {
        std::vector<unsigned char> imageData(width * height * 3); 

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = (y*width + x) * 3;  
                RGB pixelColor = pixels[y*width + x].color;
                unsigned char red = static_cast<unsigned char>(pixelColor.r);
                unsigned char green = static_cast<unsigned char>(pixelColor.g);
                unsigned char blue = static_cast<unsigned char>(pixelColor.b);

                imageData[index] = red;     
                imageData[index + 1] = green; 
                imageData[index + 2] = blue; 
            }
        }

        stbi_write_jpg(filename.c_str(), width, height, 3, imageData.data(), quality);
    }
};

// Defines a metallic material with specific ambient, diffuse, specular, and shininess properties.
class Metallic : public Material {
public:
    RGB color;
    double mA = 0.0005;
    double mD = 0.005;
    double mS = 1.0;
    double mSp = 50;
    double eta = 999999;

    Metallic(RGB _color) : color(_color) {}

    RGB getObjectColor() const override { return color; }
    double getAmbientCoefficient() const override { return mA; }
    double getDiffusionCoefficient() const override { return mD; }
    double getSpecularCoefficient() const override { return mS; }
    double getShininessCoefficient() const override { return mSp; }
    double getRefractionIndex() const override { return eta; }
};

// Defines a glassy material, characterized by high shininess and a refractive index.
class Glassy : public Material {
public:
    RGB color;
    double mA = 0.00001;
    double mD = 0.00001;
    double mS = 0.5;
    double mSp = 300;
    double eta = 1.5;

    Glassy(RGB _color) : color(_color) {}

    RGB getObjectColor() const override { return color; }
    double getAmbientCoefficient() const override { return mA; }
    double getDiffusionCoefficient() const override { return mD; }
    double getSpecularCoefficient() const override { return mS; }
    double getShininessCoefficient() const override { return mSp; }
    double getRefractionIndex() const override { return eta; }
};

// Defines a material that produces a checkerboard pattern based on world coordinates.
class CheckerboardMaterial : public Material {
public:
    RGB color1, color2;  
    double squareSize;    

    CheckerboardMaterial(RGB _color1, RGB _color2, double _squareSize)
        : color1(_color1), color2(_color2), squareSize(_squareSize) {}

    RGB getObjectColor() const override { return RGB(255, 255, 255); } 

    double getAmbientCoefficient() const override { return 1.0; }
    double getDiffusionCoefficient() const override { return 1.0; }
    double getSpecularCoefficient() const override { return 0.0; }
    double getShininessCoefficient() const override { return 0.0; }
    double getRefractionIndex() const override { return 1.0; }

    RGB getColorAtPoint(const Vec3& point) const {
        int xIndex = abs(static_cast<int>(floor(point.x / squareSize))) % 2;
        int yIndex = abs(static_cast<int>(floor(point.y / squareSize))) % 2;
        return (xIndex == yIndex) ? color1 : color2;
    }
};

// Defines a material for a soccer ball, with two panel colors and specific lighting properties.
// The color is determined by UV mapping a pattern onto the sphere.
class SimpleSoccerBallMaterial : public Material {
public:
    RGB panelColor1;
    RGB panelColor2;
    double mA, mD, mS, mSp, eta;

    SimpleSoccerBallMaterial(RGB c1, RGB c2)
        : panelColor1(c1), panelColor2(c2),
          mA(0.25), mD(0.85), mS(0.3), mSp(15.0), eta(1.0) {} 

    RGB getObjectColor() const override { return RGB(255,255,255); } 
    double getAmbientCoefficient() const override { return mA; }
    double getDiffusionCoefficient() const override { return mD; }
    double getSpecularCoefficient() const override { return mS; }
    double getShininessCoefficient() const override { return mSp; }
    double getRefractionIndex() const override { return eta; }

    RGB getColorAtPoint(const Vec3& point, const Vec3& sphereCenter) const {
        Vec3 localPoint = point - sphereCenter;
        localPoint.normalize();

        double theta = atan2(localPoint.y, localPoint.x); 
        double z_clamped = std::max(-1.0, std::min(1.0, localPoint.z));
        double phi = acos(z_clamped);                     

        double u = (theta + M_PI) / (2.0 * M_PI); 
        double v = phi / M_PI;                    

        int u_segments = 8; 
        int v_segments = 5; 

        bool u_check = (static_cast<int>(u * u_segments) % 2 == 0);
        bool v_check = (static_cast<int>(v * v_segments) % 2 == 0);

        return (u_check == v_check) ? panelColor1 : panelColor2;
    }
};

// Represents a sphere specifically for the soccer ball, using SimpleSoccerBallMaterial.
// Allows applying rotation to the texture pattern.
class SimpleSoccerBallSphere : public Sphere {
public:
    SimpleSoccerBallSphere(Vec3 _center, double _R, SimpleSoccerBallMaterial* _material)
        : Sphere(_center, _R, _material) {}

    Material* getMaterial() const override { return material; }

    RGB getColorAtPoint(const Vec3& point) const {
        SimpleSoccerBallMaterial* simpleMat = dynamic_cast<SimpleSoccerBallMaterial*>(material);
        if (simpleMat) {
            Vec3 rotatedPoint = point;
            if (currentRotationAngle > EPS && currentRotationAxis.norm() > EPS) {
                Vec3 localPoint = point - center;
                localPoint = rotate(localPoint, currentRotationAxis, -currentRotationAngle);
                rotatedPoint = center + localPoint;
            }
            return simpleMat->getColorAtPoint(rotatedPoint, center);
        }
        return RGB(255, 255, 255); 
    }
};

// Enum to select the method for calculating Fresnel reflectance.
enum ReflectionMethod {
    FRESNEL,
    SCHLICK
};

// Core ray tracing function: calculates the color seen by a ray.
// Handles object intersection, material properties, lighting (Phong), shadows, reflection, and refraction.
RGB traceRay(Ray& ray, Screen& screen, vector<Shape*>& objects, int depth, LightSource& light, RGB& ambientLight, ReflectionMethod reflMethod) {
    if (depth > MAX_DEPTH) return RGB(0, 0, 0); 

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

    if (!closestObject) return RGB(135, 206, 235); 

    if (hitMaterial) {
        if (dynamic_cast<CheckerboardMaterial*>(hitMaterial)) {
            CheckerboardMaterial* checkerboard = dynamic_cast<CheckerboardMaterial*>(hitMaterial);
            return checkerboard->getColorAtPoint(hitPoint);
        }
        
        if (dynamic_cast<SimpleSoccerBallMaterial*>(hitMaterial)) {
            SimpleSoccerBallSphere* soccerSphere = dynamic_cast<SimpleSoccerBallSphere*>(closestObject);
            if (soccerSphere) {
                Vec3 normal = closestObject->getSurfaceNormal(hitPoint);
                normal.normalize();
                
                RGB patternColor = soccerSphere->getColorAtPoint(hitPoint);
                
                Vec3 L = (light.position - hitPoint);
                L.normalize();
                Vec3 V = (ray.origin - hitPoint);
                V.normalize();
                Vec3 R_reflect = (normal * (2 * normal.dot(L))) - L; 
                R_reflect.normalize();
                
                double mA = hitMaterial->getAmbientCoefficient();
                double mD = hitMaterial->getDiffusionCoefficient();
                double mS = hitMaterial->getSpecularCoefficient();
                double mSp = hitMaterial->getShininessCoefficient();
                
                RGB ambient = patternColor * ambientLight * mA;
                RGB diffuse = patternColor * light.intensity * mD * std::max(L.dot(normal), 0.0);
                RGB specular = RGB(120, 120, 120) * light.intensity * mS * 
                               pow(std::max(V.dot(R_reflect), 0.0), mSp); 
                
                RGB phongColor = ambient + diffuse + specular; 

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
                    phongColor = ambient; 
                }
                
                phongColor.clamp();
                return phongColor;
            }
        }
    }

    Vec3 normal = closestObject->getSurfaceNormal(hitPoint);
    normal.normalize();

    RGB objectColor = hitMaterial->getObjectColor();
    double mA = hitMaterial->getAmbientCoefficient();
    double mD = hitMaterial->getDiffusionCoefficient();
    double mS = hitMaterial->getSpecularCoefficient();
    double mSp = hitMaterial->getShininessCoefficient();

    Vec3 L = (light.position - hitPoint);
    L.normalize();

    Vec3 V = (ray.origin - hitPoint);
    V.normalize();

    Vec3 R = (normal * (2 * normal.dot(L))) - L;
    R.normalize();

    RGB ambient = objectColor * ambientLight * mA;
    RGB diffuse = objectColor * light.intensity * mD * std::max(L.dot(normal), 0.0);
    RGB specular = RGB(255, 255, 255) * light.intensity * mS * pow(std::max(V.dot(R), 0.0), mSp); 

    RGB phongColor = ambient + diffuse + specular;

    Vec3 shadowOrigin = hitPoint + normal * EPS; 
    Vec3 shadowDir = (light.position - shadowOrigin);
    shadowDir.normalize();
    Ray shadowRay(shadowOrigin, shadowDir);

    double t_shadow = closestObject->solveRoot(shadowRay); 
    bool inShadow = false;
    for (Shape* obj : objects) {
        if (obj != closestObject) { 
            double t_obj_shadow = obj->solveRoot(shadowRay);
            if (t_obj_shadow > EPS && t_obj_shadow < (light.position - shadowOrigin).norm() - EPS) {
                inShadow = true;
                break;
            }
        }
    }

    if (inShadow) {
        phongColor = ambient;  
    }

    double n1 = 1.0;   
    double n2 = hitMaterial->getRefractionIndex();  

    RGB reflectionColor(0, 0, 0), refractionColor(0, 0, 0);

    if (n2 < 10) {
        bool inside = (ray.direction.dot(normal) > 0);
        if (inside) {
            std::swap(n1, n2);
            normal = -normal;
        }

        double eta = n1 / n2;
        double cosTheta1 = -normal.dot(ray.direction);
        
        double sin2Theta2 = eta * eta * (1.0 - cosTheta1 * cosTheta1);
        
        if (sin2Theta2 > 1.0) {
            Vec3 reflectionDir = ray.direction - normal * (2.0 * ray.direction.dot(normal));
            reflectionDir.normalize();
            Ray reflectedRay(hitPoint + normal * EPS, reflectionDir);
            reflectionColor = traceRay(reflectedRay, screen, objects, depth + 1, light, ambientLight, reflMethod);
        } 
        else {
            double cosTheta2 = sqrt(fabs(1.0 - sin2Theta2)); 
            Vec3 refractionDir = (ray.direction * eta) + normal * (eta * cosTheta1 - cosTheta2);
            refractionDir.normalize();
            Ray refractedRay(hitPoint - normal * EPS, refractionDir); 
            refractionColor = traceRay(refractedRay, screen, objects, depth + 1, light, ambientLight, reflMethod);
        
            Vec3 reflectionDir = ray.direction - normal * (2.0 * ray.direction.dot(normal));
            reflectionDir.normalize();
            Ray reflectedRay(hitPoint + normal * EPS, reflectionDir); 
            reflectionColor = traceRay(reflectedRay, screen, objects, depth + 1, light, ambientLight, reflMethod);
        
            double R_fresnel = 1.0; 
            double T_fresnel = 0.0;
            if (reflMethod == FRESNEL) {
                double RsBase = (n1 * cosTheta1 - n2 * cosTheta2) / (n1 * cosTheta1 + n2 * cosTheta2);
                double Rs = RsBase * RsBase;
                double RpBase = (n1 * cosTheta2 - n2 * cosTheta1) / (n1 * cosTheta2 + n2 * cosTheta1);
                double Rp = RpBase * RpBase;
                R_fresnel = Rs + Rp;
                T_fresnel = 1.0 - R_fresnel;
            }
            else if (reflMethod == SCHLICK) {
                double R0 = (n1 - n2) / (n1 + n2);
                R0 = R0 * R0;
                R_fresnel = R0 + (1.0 - R0) * pow(1.0 - cosTheta1, 5);
                T_fresnel = 1.0 - R_fresnel;
            }
        
            RGB finalColor = phongColor + (reflectionColor * R_fresnel) + (refractionColor * T_fresnel);
            finalColor.clamp();
            return finalColor;
        }
    }

    RGB tmpColor = phongColor + reflectionColor;
    tmpColor.clamp();
    return tmpColor;
}

// Processes each pixel on the screen to determine its color.
// Handles anti-aliasing by shooting multiple rays per pixel if specified.
void processScreen(Screen& screen, Vec3& origin, vector<Shape*>& objects, LightSource& light, RGB& ambientLight, ANTI_ALIASING aa, ReflectionMethod reflMethod) {
    #pragma omp parallel for
    for (auto& pixel : screen) {
        double aspectRatio = (double)screen.width / screen.height;
        stack<Ray> st;

        auto computePixelPosition = [&](double px, double py) -> Vec3 {
            Vec3 screenCenter = origin + screen.normal * screen.pov;
            Vec3 pixelPosition = screenCenter 
                + screen.right * ((px - screen.width / 2.0) / screen.width * aspectRatio) 
                + screen.up * ((screen.height / 2.0 - py) / screen.height); 
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

        int rayCount = st.size();
        RGB accumulatedColor(0, 0, 0);

        while (!st.empty()) {
            Ray ray = st.top();
            st.pop();

            RGB color = traceRay(ray, screen, objects, 0, light, ambientLight, reflMethod);
            accumulatedColor += color;
        }

        pixel.color = accumulatedColor / rayCount;
        pixel.color.clamp(); 
    }
}  

// Loads a 3D model from an OBJ file.
// Parses vertices and faces, calculates vertex normals, and creates Triangle objects.
vector<Shape*> loadOBJ(const string& filename, Material* material) {
    vector<Vec3> vertices;
    vector<Vec3> vertexNormals; 
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
            vertexNormals.emplace_back(0, 0, 0); 
        } else if (prefix == "f") {
            int v1, v2, v3;
            iss >> v1 >> v2 >> v3;
            faces.emplace_back(v1 - 1, v2 - 1, v3 - 1); 
        }
    }

    for (const auto& [i1, i2, i3] : faces) {
        Vec3& p1 = vertices[i1];
        Vec3& p2 = vertices[i2];
        Vec3& p3 = vertices[i3];

        Vec3 faceNormal = (p2 - p1).cross(p3 - p1);

        vertexNormals[i1] = vertexNormals[i1] + faceNormal;
        vertexNormals[i2] = vertexNormals[i2] + faceNormal;
        vertexNormals[i3] = vertexNormals[i3] + faceNormal;
    }

    for (auto& n : vertexNormals) {
        n.normalize();
    }

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

// Helper function to create a cuboid (represented by 12 triangles) and add it to the scene.
void addCuboidToObjects(Vec3 center, Vec3 size, Material* mat, std::vector<Shape*>& shapes) {
    double dx = size.x / 2.0;
    double dy = size.y / 2.0;
    double dz = size.z / 2.0;

    Vec3 v[8];
    v[0] = center + Vec3(-dx, -dy, -dz); 
    v[1] = center + Vec3(+dx, -dy, -dz); 
    v[2] = center + Vec3(+dx, +dy, -dz); 
    v[3] = center + Vec3(-dx, +dy, -dz); 
    v[4] = center + Vec3(-dx, -dy, +dz); 
    v[5] = center + Vec3(+dx, -dy, +dz); 
    v[6] = center + Vec3(+dx, +dy, +dz); 
    v[7] = center + Vec3(-dx, +dy, +dz); 

    shapes.push_back(new Triangle(v[0], v[3], v[2], mat));
    shapes.push_back(new Triangle(v[0], v[2], v[1], mat));
    shapes.push_back(new Triangle(v[4], v[5], v[6], mat));
    shapes.push_back(new Triangle(v[4], v[6], v[7], mat));
    shapes.push_back(new Triangle(v[0], v[1], v[5], mat));
    shapes.push_back(new Triangle(v[0], v[5], v[4], mat));
    shapes.push_back(new Triangle(v[3], v[7], v[6], mat));
    shapes.push_back(new Triangle(v[3], v[6], v[2], mat));
    shapes.push_back(new Triangle(v[0], v[4], v[7], mat));
    shapes.push_back(new Triangle(v[0], v[7], v[3], mat));
    shapes.push_back(new Triangle(v[1], v[2], v[6], mat));
    shapes.push_back(new Triangle(v[1], v[6], v[5], mat));
}

// Loads ball waypoints, spin data, goal, and wall objects from a CSV file.
// Transforms coordinates from simulation space to world space and creates corresponding shapes.
void loadWaypointsAndObjectsFromCSV(
    const std::string& filename,
    std::vector<Vec3>& ballWaypoints_simCoords, 
    std::vector<Vec3>& ballSpins_simCoords,     
    std::vector<Shape*>& objects,               
    Material* sceneMaterial,                     
    Vec3& goalPosition_simCoords                
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

    bool goal_added = false; // Flag to ensure goal is added only once

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

                if (!goal_added) {
                    double goal_width_sim = std::stod(segments[4]);  // e.g., 7.32m
                    double goal_height_sim = std::stod(segments[5]); // e.g., 2.44m
                    // segments[6-8] is orientation, assume (1,0,0) i.e. faces +X sim axis

                    // Convert goal properties to world coordinates
                    Vec3 goal_ref_point_world; // Center of goal line on the ground
                    goal_ref_point_world.x = csv_x;       // Sim X (depth) -> World X
                    goal_ref_point_world.y = csv_z;       // Sim Z (width) -> World Y
                    goal_ref_point_world.z = csv_y;       // Sim Y (height) -> World Z (ground level)

                    double post_thickness = 0.12; // FIFA standard max 12cm

                    // World dimensions
                    double world_goal_width_internal = goal_width_sim; // Along World Y
                    double world_goal_height_internal = goal_height_sim; // Along World Z

                    // Left Post
                    Vec3 lp_center;
                    lp_center.x = goal_ref_point_world.x;
                    lp_center.y = goal_ref_point_world.y - (world_goal_width_internal / 2.0) - (post_thickness / 2.0);
                    lp_center.z = goal_ref_point_world.z + (world_goal_height_internal / 2.0);
                    Vec3 lp_size(post_thickness, post_thickness, world_goal_height_internal);
                    addCuboidToObjects(lp_center, lp_size, sceneMaterial, objects);

                    // Right Post
                    Vec3 rp_center;
                    rp_center.x = goal_ref_point_world.x;
                    rp_center.y = goal_ref_point_world.y + (world_goal_width_internal / 2.0) + (post_thickness / 2.0);
                    rp_center.z = goal_ref_point_world.z + (world_goal_height_internal / 2.0);
                    Vec3 rp_size(post_thickness, post_thickness, world_goal_height_internal);
                    addCuboidToObjects(rp_center, rp_size, sceneMaterial, objects);
                    
                    // Crossbar
                    Vec3 cb_center;
                    cb_center.x = goal_ref_point_world.x;
                    cb_center.y = goal_ref_point_world.y;
                    cb_center.z = goal_ref_point_world.z + world_goal_height_internal + (post_thickness / 2.0);
                    // Length of crossbar spans from center of left post to center of right post
                    double crossbar_length_world_y = world_goal_width_internal + post_thickness;
                    Vec3 cb_size(post_thickness, crossbar_length_world_y, post_thickness);
                    addCuboidToObjects(cb_center, cb_size, sceneMaterial, objects);
                    
                    goal_added = true;
                }

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
                Triangle* tri1 = new Triangle(v0, v1, v2, sceneMaterial);
                Triangle* tri2 = new Triangle(v0, v2, v3, sceneMaterial);
                
                objects.push_back(tri1);
                objects.push_back(tri2);
            }
        } catch (const std::invalid_argument& ia) {
        } catch (const std::out_of_range& oor) {
        }
    }
    file.close();
}

// Defines a basic 5x7 pixel font for text rendering.
const std::map<char, std::vector<std::string>> FONT_DATA = {
    {'0', {"01110", "01010", "01010", "01010", "01010", "01010", "01110"}},
    {'1', {"00100", "01100", "00100", "00100", "00100", "00100", "01110"}},
    {'2', {"01110", "00010", "00010", "01110", "01000", "01000", "01110"}},
    {'3', {"01110", "00010", "00010", "00110", "00010", "00010", "01110"}},
    {'4', {"01010", "01010", "01010", "01110", "00010", "00010", "00010"}},
    {'5', {"01110", "01000", "01000", "01110", "00010", "00010", "01110"}},
    {'6', {"01110", "01000", "01000", "01110", "01010", "01010", "01110"}},
    {'7', {"01110", "00010", "00010", "00100", "00100", "00100", "00100"}},
    {'8', {"01110", "01010", "01010", "01110", "01010", "01010", "01110"}},
    {'9', {"01110", "01010", "01010", "01110", "00010", "00010", "01110"}},
    {'.', {"00000", "00000", "00000", "00000", "00000", "00100", "00100"}},
    {':', {"00000", "00100", "00100", "00000", "00100", "00100", "00000"}},
    {' ', {"00000", "00000", "00000", "00000", "00000", "00000", "00000"}},
    {'s', {"00000", "01110", "01000", "01110", "00010", "01110", "00000"}},
    {'t', {"00100", "00100", "01110", "00100", "00100", "00100", "00110"}},
    {'T', {"01110", "00100", "00100", "00100", "00100", "00100", "00100"}},
    {'i', {"00100", "00000", "00100", "00100", "00100", "00100", "00100"}},
    {'m', {"00000", "01010", "11111", "10101", "10101", "10001", "00000"}},
    {'e', {"00000", "01110", "01010", "01110", "01000", "01110", "00000"}},
    {'o', {"00000", "01110", "01010", "01010", "01010", "01110", "00000"}},
    {'G', {"01110", "01000", "01000", "01011", "01010", "01010", "01110"}},
    {'a', {"00000", "01110", "00010", "01110", "01010", "01110", "00000"}},
    {'l', {"00100", "00100", "00100", "00100", "00100", "00100", "00110"}}
};

const int FONT_CHAR_WIDTH = 5;
const int FONT_CHAR_HEIGHT = 7;
const int FONT_CHAR_SPACING = 1;

// Draws a single character onto the screen's pixel buffer at a specified location, color, and scale.
void drawChar(std::vector<Pixel>& pixels, int screen_width, int screen_height,
              char c, int start_x, int start_y, RGB color, int scale) { 
    auto it = FONT_DATA.find(c);
    if (it == FONT_DATA.end()) return; 

    const auto& char_pattern = it->second;
    
    for (int r = 0; r < FONT_CHAR_HEIGHT; ++r) {
        if (r >= char_pattern.size()) continue;
        const std::string& row_pattern = char_pattern[r];
        for (int col = 0; col < FONT_CHAR_WIDTH; ++col) {
            if (col >= row_pattern.length()) continue;
            if (row_pattern[col] == '1' || row_pattern[col] == '#') {
                for (int sr = 0; sr < scale; ++sr) { 
                    for (int sc = 0; sc < scale; ++sc) {
                        int px = start_x + col * scale + sc;
                        int py = start_y + r * scale + sr;
                        if (px >= 0 && px < screen_width && py >= 0 && py < screen_height) {
                            pixels[py * screen_width + px].color = color;
                        }
                    }
                }
            }
        }
    }
}

// Draws a string of text onto the screen using the drawChar function.
void drawText(Screen& screen, const std::string& text_to_draw,
              int x, int y, RGB color, int scale) { 
    int current_x = x;
    for (char c : text_to_draw) {
        drawChar(screen.pixels, screen.width, screen.height, c, current_x, y, color, scale);
        current_x += (FONT_CHAR_WIDTH + FONT_CHAR_SPACING) * scale; 
    }
}

// Renders a sequence of frames depicting a moving soccer ball.
// For each frame, it updates the ball's position and spin, sets up the camera,
// renders the scene, draws overlay text (time to goal), and saves the image.
void renderMovingSphere(const std::vector<Vec3>& ballWaypoints_simCoords, 
                        const std::vector<Vec3>& ballSpins_simCoords,
                        Vec3 camPos, 
                        std::vector<Shape*>& objects, 
                        SimpleSoccerBallMaterial* ballMaterial,  
                        LightSource& light, 
                        RGB ambientLight, 
                        ANTI_ALIASING aa, 
                        ReflectionMethod reflMethod,
                        const std::string& outputDir) { 
    
    SimpleSoccerBallSphere* movingSphere = new SimpleSoccerBallSphere(Vec3(0,0,0), 0.11, ballMaterial);
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
        movingSphere->center.x = waypoint_sim.x; 
        movingSphere->center.y = waypoint_sim.z; 
        movingSphere->center.z = waypoint_sim.y; 

        if (i < ballSpins_simCoords.size()) {
            const Vec3& spin_sim = ballSpins_simCoords[i];
            Vec3 spin_world;
            spin_world.x = spin_sim.x; 
            spin_world.y = spin_sim.z; 
            spin_world.z = spin_sim.y; 

            double rot_angle = spin_world.norm();
            Vec3 rot_axis_world = spin_world;

            if (rot_angle > EPS) {
                rot_axis_world = rot_axis_world / rot_angle; 
            } else {
                rot_axis_world = Vec3(0, 0, 1); 
                rot_angle = 0.0;
            }
            movingSphere->currentRotationAxis = rot_axis_world;
            movingSphere->currentRotationAngle = rot_angle; 
        }
        
        Vec3 screenNormal = movingSphere->center - camPos;
        if (screenNormal.norm() < EPS) {
            screenNormal = Vec3(1,0,0); 
        } else {
            screenNormal.normalize();
        }
        
        Screen screen(screenNormal, 2560, 1440);
        screen.pov = 0.9; 
        processScreen(screen, camPos, objects, light, ambientLight, aa, reflMethod);

        double time_per_frame = 0.1; 
        double time_to_goal = (numFrames - 1 - i) * time_per_frame;
        if (time_to_goal < 0.0) time_to_goal = 0.0; 

        std::ostringstream time_text_stream;
        time_text_stream << std::fixed << std::setprecision(1) << "Time to Goal: " << time_to_goal << " s";
        std::string time_text = time_text_stream.str();
        
        int font_scale = 4; 
        int text_margin_x = 10 * font_scale; 
        int text_margin_y = 10 * font_scale;
        int text_start_x = text_margin_x;
        int text_start_y = screen.height - (FONT_CHAR_HEIGHT * font_scale) - text_margin_y; 
        RGB text_color(255, 255, 255); 

        drawText(screen, time_text, text_start_x, text_start_y, text_color, font_scale);

        std::ostringstream filenameStream;
        filenameStream << outputDir << "/frame_" << std::setw(4) << std::setfill('0') << i << ".jpg"; 
        screen.writeToJPG(filenameStream.str());
    }
    objects.erase(std::remove(objects.begin(), objects.end(), movingSphere), objects.end());
    delete movingSphere;
}

// Main function: Initializes the scene, processes CSV files for ball trajectories and obstacles,
// and renders an animation for each trajectory.
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Not enough parameters. Usage: <program> <ReflectionMethod> <MAX_DEPTH>" << endl;
        exit(-1);
    }
    ReflectionMethod reflMethod = (ReflectionMethod)atoi(argv[1]);
    MAX_DEPTH = (int)atoi(argv[2]);

    std::string baseInputDir = "../01_KickSimulator/output/";
    std::string baseOutputDir = "./output/";

    if (!std::filesystem::exists(baseOutputDir)) {
        std::filesystem::create_directories(baseOutputDir);
    }
    
    Vec3 origin(0.0, 0.0, 0.0); 
    double radius = 3.0;  
    double glassRadius = radius;  
    double smallSphereRadius = 0.25 * 2;  
    Vec3 glassCenter(10, 0, 0);  

    Glassy glassy(RGB(255, 255, 255));
    Metallic redMetal(RGB(255, 0, 0));
    Metallic blueMetal(RGB(0, 0, 255));
    Metallic greenMetal(RGB(0, 255, 0));
    Metallic yellowMetal(RGB(255, 255, 0));
    Metallic whiteMetal(RGB(255, 255, 255));
    Metallic darkGrayMetal(RGB(50, 50, 50)); 
    CheckerboardMaterial checkerboard(RGB(34, 139, 34), RGB(0, 100, 0), radius); 
    
    SimpleSoccerBallMaterial soccerBallMat( 
        RGB(15, 15, 15),    
        RGB(230, 230, 230)  
    );

    std::vector<Shape*> objects; 

    Plane floor(Vec3(0, 0, -1), Vec3(0, 0, 1), &checkerboard);

    LightSource light(Vec3(5.0, -10.0, 10.0), RGB(220, 220, 220)); 
    RGB ambientLight(85, 85, 85); 

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

                size_t objects_count_before_load = objects.size();
                Vec3 goalPosition_simCoords(0,0,0); 
                loadWaypointsAndObjectsFromCSV(currentWavepointFile, 
                                               ballWaypoints_simCoords, ballSpins_simCoords, objects, &darkGrayMetal, goalPosition_simCoords); 
                size_t objects_count_after_load = objects.size();

                for (size_t i = objects_count_before_load; i < objects_count_after_load; ++i) {
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
                    }
                }
    
                renderMovingSphere(ballWaypoints_simCoords, ballSpins_simCoords, goalkeeper_pos_world, 
                                   objects, &soccerBallMat, light, ambientLight, AA_REGULAR_4, reflMethod, currentOutputDir);

                for (Shape* loaded_shape : csv_wall_triangles_managed) {
                    objects.erase(std::remove(objects.begin(), objects.end(), loaded_shape), objects.end());
                    delete loaded_shape;
                }
                csv_wall_triangles_managed.clear();
                
                std::cout << "Finished processing for " << filename << ". Output in " << currentOutputDir << std::endl;
            }
        }
    }

    return 0;
}