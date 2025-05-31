// ---------------------------------------------------------------------------
// main.cpp — Soccer‐ball free‐flight & bounce demo (PBR grass + ImGui +
//               4-person "wall" + goal)
// ---------------------------------------------------------------------------

#include <iostream>
#include <cmath>
#define _USE_MATH_DEFINES  // ensure M_PI on MSVC

#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Dear ImGui
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// stb_image
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// TinyOBJLoader
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// 1. GLSL Shaders
// ---------------------------------------------------------------------------

static const char* VERT_SHADER = R"(
#version 330 core
layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inNorm;
layout(location=2) in vec2 inUV;

uniform mat4 model, view, proj;

out vec3 vPos;
out vec3 vNorm;
out vec2 vUv;

void main()
{
    vec4 world = model * vec4(inPos, 1.0);
    vPos  = world.xyz;
    vNorm = mat3(transpose(inverse(model))) * inNorm;
    vUv   = inUV;
    gl_Position = proj * view * world;
}
)";

static const char* FRAG_SHADER = R"(
#version 330 core
in  vec3 vPos;
in  vec3 vNorm;
in  vec2 vUv;
out vec4 FragColor;

uniform vec3  lightPos;
uniform vec3  camPos;
uniform vec3  lightColor;
uniform float roughness;
uniform float metallic;
uniform float exposure;

// 0 = grass, 1 = ball, 2 = goal, 3 = wall
uniform int objectType;

// Grass textures
uniform sampler2D grass_albedo;
uniform sampler2D grass_normal;
uniform sampler2D grass_roughness;

// Ball textures
uniform sampler2D ball_albedo;
uniform sampler2D ball_normal;
uniform sampler2D ball_roughness;

// Goal textures
uniform sampler2D goal_albedo;
uniform sampler2D goal_normal;
uniform sampler2D goal_roughness;

// Wall textures
uniform sampler2D wall_albedo;
uniform sampler2D wall_normal;
uniform sampler2D wall_roughness;

// --------------------------------------------------------------------
// PBR shading
// --------------------------------------------------------------------
float D_GGX(vec3 N, vec3 H, float a)
{
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    return a2 / (3.14159 * denom * denom);
}
float G_Schlick(float NdotV, float k)
{
    return NdotV / (NdotV * (1.0 - k) + k);
}
float G_Smith(vec3 N, vec3 V, vec3 L, float a)
{
    float k = (a + 1.0);
    k = (k * k) / 8.0;
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return G_Schlick(NdotV, k) * G_Schlick(NdotL, k);
}
vec3 F_Schlick(float cosT, vec3 F0)
{
    return F0 + (1.0 - F0) * pow((1.0 - cosT), 5.0);
}

void main()
{
    vec3 N = normalize(vNorm);
    vec3 V = normalize(camPos - vPos);
    vec3 L = normalize(lightPos - vPos);
    vec3 H = normalize(V + L);

    vec3 baseColor;
    float rVal;
    float mVal;
    vec3 nMap;

    if(objectType == 0)
    {
        // grass
        baseColor = texture(grass_albedo, vUv * 50.0).rgb;
        nMap      = texture(grass_normal, vUv * 50.0).rgb * 2.0 - 1.0;
        rVal      = texture(grass_roughness, vUv * 50.0).r;
        mVal      = 0.0;
    }
    else if(objectType == 1)
    {
        // ball
        baseColor = texture(ball_albedo, vUv).rgb;
        nMap      = texture(ball_normal, vUv).rgb * 2.0 - 1.0;
        rVal      = texture(ball_roughness, vUv).r;
        mVal      = 0.0;
    }
    else if(objectType == 2)
    {
        // goal
        baseColor = texture(goal_albedo, vUv).rgb;
        nMap      = texture(goal_normal, vUv).rgb * 2.0 - 1.0;
        rVal      = texture(goal_roughness, vUv).r;
        mVal      = 0.0;
    }
    else
    {
        // wall (4 persons)
        baseColor = texture(wall_albedo, vUv).rgb;
        nMap      = texture(wall_normal, vUv).rgb * 2.0 - 1.0;
        rVal      = texture(wall_roughness, vUv).r;
        mVal      = 0.0;
    }

    // mix user-defined roughness/metallic
    rVal = mix(rVal, roughness, 0.5);
    mVal = mix(mVal, metallic,  0.2);

    // combine normal
    N = normalize(N + nMap * 0.5);

    // GGX
    vec3 F0 = mix(vec3(0.04), baseColor, mVal);
    vec3 F  = F_Schlick(max(dot(H, V), 0.0), F0);
    float D = D_GGX(N, H, rVal);
    float G = G_Smith(N, V, L, rVal);

    vec3 specNumer = (D * G * F);
    float denom    = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specularTerm = specNumer / denom;

    vec3 kd = (1.0 - F) * (1.0 - mVal);
    vec3 diffuseTerm = kd * baseColor / 3.14159;

    vec3 colorOut = (diffuseTerm + specularTerm) * max(dot(N, L), 0.0) * lightColor;
    colorOut = exposure * (0.2 * baseColor + colorOut);

    FragColor = vec4(colorOut, 1.0);
}
)";

// ---------------------------------------------------------------------------
// 2. utility helpers
// ---------------------------------------------------------------------------

GLuint compile(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok; 
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if(!ok)
    {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "Shader error:\n" << log << "\n";
    }
    return s;
}

GLuint loadTex(const char* path)
{
    stbi_set_flip_vertically_on_load(true);
    int w, h, c;
    unsigned char* d = stbi_load(path, &w, &h, &c, 0);
    if(!d)
    {
        std::cerr << "Texture fail: " << path << "\n";
        return 0;
    }
    GLenum fmt = (c == 1) ? GL_RED : (c == 3) ? GL_RGB : GL_RGBA;

    GLuint id;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexImage2D(GL_TEXTURE_2D, 0, fmt, w, h, 0, fmt, GL_UNSIGNED_BYTE, d);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(d);
    return id;
}

struct MeshGL
{
    GLuint vao;
    GLuint vbo;
    GLuint ebo;
    GLsizei indexCount;
};

bool loadObjMesh(const std::string& filename,
                 const std::string& mtlDir,
                 MeshGL& outMesh)
{
    tinyobj::attrib_t                attrib;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ret = tinyobj::LoadObj(
        &attrib, &shapes, &materials,
        &warn, &err,
        filename.c_str(),
        mtlDir.empty() ? nullptr : mtlDir.c_str()
    );
    if(!warn.empty())  std::cout << "TinyOBJ warning: " << warn << "\n";
    if(!err.empty())   std::cerr << "TinyOBJ error:   " << err << "\n";
    if(!ret) return false;
    if(shapes.empty()) {
        std::cerr << "Error: no shapes in " << filename << "\n";
        return false;
    }

    std::vector<float> vertexData;
    std::vector<unsigned int> indexData;
    size_t indexOffset = 0;

    for(const auto& shape : shapes)
    {
        for(size_t f = 0; f < shape.mesh.indices.size(); ++f)
        {
            tinyobj::index_t idx = shape.mesh.indices[f];

            float px = attrib.vertices[3*idx.vertex_index + 0];
            float py = attrib.vertices[3*idx.vertex_index + 1];
            float pz = attrib.vertices[3*idx.vertex_index + 2];

            float nx = 0.0f, ny = 1.0f, nz = 0.0f;
            if(idx.normal_index >= 0) {
                nx = attrib.normals[3*idx.normal_index + 0];
                ny = attrib.normals[3*idx.normal_index + 1];
                nz = attrib.normals[3*idx.normal_index + 2];
            }
            float u = 0.0f, v = 0.0f;
            if(idx.texcoord_index >= 0) {
                u = attrib.texcoords[2*idx.texcoord_index + 0];
                v = attrib.texcoords[2*idx.texcoord_index + 1];
            }

            vertexData.push_back(px);
            vertexData.push_back(py);
            vertexData.push_back(pz);
            vertexData.push_back(nx);
            vertexData.push_back(ny);
            vertexData.push_back(nz);
            vertexData.push_back(u);
            vertexData.push_back(v);

            indexData.push_back(static_cast<unsigned int>(indexOffset + f));
        }
        indexOffset += shape.mesh.indices.size();
    }

    glGenVertexArrays(1, &outMesh.vao);
    glGenBuffers(1, &outMesh.vbo);
    glGenBuffers(1, &outMesh.ebo);

    glBindVertexArray(outMesh.vao);

    glBindBuffer(GL_ARRAY_BUFFER, outMesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float),
                 vertexData.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, outMesh.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexData.size() * sizeof(unsigned int),
                 indexData.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);

    outMesh.indexCount = (GLsizei)indexData.size();
    return true;
}

// ---------------------------------------------------------------------------
// 3. Globals & physics
// ---------------------------------------------------------------------------
float       g_gravity     = -9.81f;
glm::vec3   g_ballPos(0.0f);
glm::vec3   g_v0(-2.0f, 10.0f, -45.0f);
bool        g_followCam   = false;
float       g_rough       = 0.3f,  g_metal = 0.0f;
float       g_exposure    = 1.5f;

const float BALL_SCALE    = 0.5f;
const float BALL_RADIUS   = 0.5f;

float       g_restitution = 0.7f;
float       g_frictionXZ  = 0.8f;

glm::vec3   g_spinRadSec  = glm::vec3(0.0f, 80.0f, 0.0f);
float       g_magnusK     = 6e-4f;
float       g_spinDecay   = 0.1f;
bool        g_dragEnable  = true;
float       g_dragCoeff   = 3e-4f;

float       g_squashRecover = 5.0f;

// goal
float       g_goalZ = -40.0f;

// goal bars
struct Bar {
    glm::vec3 center;
    glm::vec3 halfsize;
};
Bar g_topBar   = { glm::vec3(0.0f,  2.4f, -40.0f), glm::vec3(3.65f, 0.05f, 0.05f) };
Bar g_leftBar  = { glm::vec3(-3.65f, 1.2f, -40.0f), glm::vec3(0.05f, 1.2f, 0.05f) };
Bar g_rightBar = { glm::vec3( 3.65f, 1.2f, -40.0f), glm::vec3(0.05f, 1.2f, 0.05f) };

// For wall model bounding box
glm::vec3 g_wallVMin( 1e9f);
glm::vec3 g_wallVMax(-1e9f);
float     g_wallScale = 1.0f;

// ---------------------------------------------------------------------------
// 4. collision: ball vs bar
// ---------------------------------------------------------------------------
bool checkBarCollision(const glm::vec3& ballCenter,
                       float radius,
                       const Bar& bar,
                       glm::vec3& normalOut)
{
    glm::vec3 p = glm::clamp(ballCenter,
                             bar.center - bar.halfsize,
                             bar.center + bar.halfsize);
    glm::vec3 diff = p - ballCenter;
    float dist2 = glm::dot(diff, diff);
    if(dist2 < radius*radius)
    {
        float d = sqrt(dist2);
        if(d < 1e-5f) {
            normalOut = glm::vec3(1,0,0);
        } else {
            normalOut = diff / d; 
        }
        return true;
    }
    return false;
}
void bounceBall(glm::vec3& vel, const glm::vec3& normal, float restitution)
{
    float vn = glm::dot(vel, normal);
    if(vn < 0.0f) {
        vel = vel - (1.0f + restitution)*vn*normal;
    }
}

// ---------------------------------------------------------------------------
// 5. buildPlane
// ---------------------------------------------------------------------------
MeshGL g_planeMesh;
void buildPlane(MeshGL& mesh, float s = 100.0f)
{
    float vb[] = {
        -s, 0, -s,    0,1,0,   0,0,
         s, 0, -s,    0,1,0,   1,0,
         s, 0,  s,    0,1,0,   1,1,
        -s, 0,  s,    0,1,0,   0,1
    };
    unsigned int ib[] = {0,1,2, 2,3,0};

    glGenVertexArrays(1, &mesh.vao);
    glGenBuffers(1, &mesh.vbo);
    glGenBuffers(1, &mesh.ebo);

    glBindVertexArray(mesh.vao);
      glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
      glBufferData(GL_ARRAY_BUFFER, sizeof(vb), vb, GL_STATIC_DRAW);

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ib), ib, GL_STATIC_DRAW);

      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)0);
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
      glEnableVertexAttribArray(1);
      glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(6*sizeof(float)));
      glEnableVertexAttribArray(2);

    glBindVertexArray(0);

    mesh.indexCount = 6;
}

// ---------------------------------------------------------------------------
// 6. main
// ---------------------------------------------------------------------------
int main()
{
    // GLFW / OpenGL init
    if(!glfwInit()) {
        std::cerr << "glfw init failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    GLFWwindow* win = glfwCreateWindow(1280,720,"Soccer Demo w/ Wall",nullptr,nullptr);
    if(!win) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(win);
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n";
        return -1;
    }
    glfwSwapInterval(1);

    // ImGui init
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(win,true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // compile/link shaders
    GLuint vs   = compile(GL_VERTEX_SHADER,   VERT_SHADER);
    GLuint fs   = compile(GL_FRAGMENT_SHADER, FRAG_SHADER);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    // build plane
    buildPlane(g_planeMesh);

    // load ball
    MeshGL ballMesh;
    if(!loadObjMesh("assets/soccer_ball.obj","assets/", ballMesh)) {
        std::cerr<<"Failed to load soccer_ball.obj\n";
        return -1;
    }

    // load goal
    MeshGL goalMesh;
    if(!loadObjMesh("assets/goal.obj","assets/", goalMesh)) {
        std::cerr<<"Failed to load goal.obj\n";
        return -1;
    }

    // load man (single person)
    MeshGL wallMesh;
    if(!loadObjMesh("assets/Man.obj","assets/", wallMesh)) {
        std::cerr<<"Failed to load Man.obj\n";
        return -1;
    }

    // compute bounding box for the wall model, to scale to ~1.8m tall
    {
        glBindBuffer(GL_ARRAY_BUFFER, wallMesh.vbo);
        GLint bufBytes=0;
        glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bufBytes);
        size_t vertCount = size_t(bufBytes)/(8*sizeof(float)); // 8 floats/vertex

        const float* vb = (const float*) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
        if(vb)
        {
            // reset
            g_wallVMin = glm::vec3( 1e9f);
            g_wallVMax = glm::vec3(-1e9f);

            for(size_t i=0;i<vertCount;++i)
            {
                glm::vec3 p(vb[i*8+0], vb[i*8+1], vb[i*8+2]);
                g_wallVMin = glm::min(g_wallVMin, p);
                g_wallVMax = glm::max(g_wallVMax, p);
            }
            glUnmapBuffer(GL_ARRAY_BUFFER);

            float modelHeight = g_wallVMax.y - g_wallVMin.y;
            if(modelHeight>1e-5f)
            {
                float targetH = 1.8f;  // want ~1.8m tall
                g_wallScale   = targetH / modelHeight;
            }
        }
    }

    // load textures
    GLuint grassDiff=loadTex("assets/grass_diffuse.png");
    GLuint grassNorm=loadTex("assets/grass_normal.png");
    GLuint grassRgh =loadTex("assets/grass_roughness.png");

    GLuint ballDiff=loadTex("assets/ball_diffuse.jpg");
    GLuint ballNorm=loadTex("assets/ball_normal.jpg");
    GLuint ballRgh =loadTex("assets/ball_roughness.jpg");

    GLuint goalDiff=loadTex("assets/goal_diffuse.png");
    GLuint goalNorm=loadTex("assets/goal_normal.png");
    GLuint goalRgh =loadTex("assets/goal_roughness.png");

    // for the man
    GLuint wallDiff=loadTex("assets/Man_BaseColor.png");
    GLuint wallNorm=loadTex("assets/Man_Normal.png");
    GLuint wallRgh =loadTex("assets/Man_Roughness.png");

    // uniforms
    glUseProgram(prog);
    GLint uM    = glGetUniformLocation(prog,"model");
    GLint uV    = glGetUniformLocation(prog,"view");
    GLint uP    = glGetUniformLocation(prog,"proj");
    GLint uLP   = glGetUniformLocation(prog,"lightPos");
    GLint uLC   = glGetUniformLocation(prog,"lightColor");
    GLint uCP   = glGetUniformLocation(prog,"camPos");
    GLint uR    = glGetUniformLocation(prog,"roughness");
    GLint uMet  = glGetUniformLocation(prog,"metallic");
    GLint uExp  = glGetUniformLocation(prog,"exposure");
    GLint uType = glGetUniformLocation(prog,"objectType");

    glUniform3f(uLP,0.0f,30.0f,30.0f);
    glUniform3f(uLC,3.0f,3.0f,3.0f);

    // set sampler2D units
    glUniform1i(glGetUniformLocation(prog,"grass_albedo"),    0);
    glUniform1i(glGetUniformLocation(prog,"grass_normal"),    1);
    glUniform1i(glGetUniformLocation(prog,"grass_roughness"), 2);

    glUniform1i(glGetUniformLocation(prog,"ball_albedo"),     3);
    glUniform1i(glGetUniformLocation(prog,"ball_normal"),     4);
    glUniform1i(glGetUniformLocation(prog,"ball_roughness"),  5);

    glUniform1i(glGetUniformLocation(prog,"goal_albedo"),     6);
    glUniform1i(glGetUniformLocation(prog,"goal_normal"),     7);
    glUniform1i(glGetUniformLocation(prog,"goal_roughness"),  8);

    glUniform1i(glGetUniformLocation(prog,"wall_albedo"),     9);
    glUniform1i(glGetUniformLocation(prog,"wall_normal"),     10);
    glUniform1i(glGetUniformLocation(prog,"wall_roughness"),  11);

    // projection
    int wScreen, hScreen;
    glfwGetFramebufferSize(win,&wScreen,&hScreen);
    glm::mat4 projMat = glm::perspective(glm::radians(45.0f),
                         float(wScreen)/float(hScreen), 0.1f, 500.0f);
    glUniformMatrix4fv(uP,1,GL_FALSE,glm::value_ptr(projMat));

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    double lastTime = glfwGetTime();
    float spinAngle = 0.0f;

    while(!glfwWindowShouldClose(win))
    {
        double now=glfwGetTime();
        float dt=float(now-lastTime);
        lastTime=now;

        glfwPollEvents();

        // physics
        glm::vec3 vel = g_v0;

        glm::vec3 acc(0.0f,g_gravity,0.0f);

        // magnus
        glm::vec3 magnus = g_magnusK*glm::cross(g_spinRadSec, vel);
        acc += magnus;

        // drag
        if(g_dragEnable)
        {
            float spd=glm::length(vel);
            if(spd>1e-3f) acc -= g_dragCoeff*spd*vel;
        }

        // integration
        g_v0 += acc*dt;
        g_ballPos += g_v0*dt;

        // spin decay
        g_spinRadSec*=exp(-g_spinDecay*dt);

        // ground bounce
        {
            float bottom=g_ballPos.y - BALL_RADIUS;
            if(bottom<0.0f)
            {
                g_ballPos.y -= bottom;
                if(g_v0.y<0.0f) g_v0.y = -g_v0.y*g_restitution;
                g_v0.x*=g_frictionXZ;
                g_v0.z*=g_frictionXZ;
                if(fabs(g_v0.y)<0.05f) g_v0.y=0.0f;
                if(glm::length(glm::vec2(g_v0.x,g_v0.z))<0.05f)
                {
                    g_v0.x=0.0f; 
                    g_v0.z=0.0f;
                }
            }
        }

        // goal bar collision
        {
            glm::vec3 nrm;
            if(checkBarCollision(g_ballPos,BALL_RADIUS,g_topBar,nrm))   bounceBall(g_v0,nrm,g_restitution);
            if(checkBarCollision(g_ballPos,BALL_RADIUS,g_leftBar,nrm))  bounceBall(g_v0,nrm,g_restitution);
            if(checkBarCollision(g_ballPos,BALL_RADIUS,g_rightBar,nrm)) bounceBall(g_v0,nrm,g_restitution);
        }

        // goal detect
        if(g_ballPos.z<g_goalZ+1.0f && g_ballPos.z>g_goalZ-1.5f &&
           fabs(g_ballPos.x)<3.65f &&
           g_ballPos.y<2.5f &&
           glm::length(g_v0)<2.0f)
        {
            g_v0=glm::vec3(0.0f);
        }

        spinAngle+=5.0f*dt;

        // ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Controls");
        ImGui::Checkbox("Follow Camera",&g_followCam);
        if(ImGui::Button("Reset"))
        {
            g_ballPos=glm::vec3(0.0f);
            g_v0=glm::vec3(-2.0f,10.0f,-45.0f);
        }
        ImGui::SliderFloat3("Velocity",&g_v0.x,-50.0f,50.0f);
        ImGui::SliderFloat("Restitution",&g_restitution,0.0f,1.0f);
        ImGui::SliderFloat("FrictionXZ",&g_frictionXZ,0.0f,1.0f);
        ImGui::SliderFloat("Exposure",&g_exposure,0.1f,5.0f);
        ImGui::SliderFloat("Roughness",&g_rough,0.0f,1.0f);
        ImGui::SliderFloat("Metallic",&g_metal,0.0f,1.0f);

        ImGui::SeparatorText("Spin / Magnus");
        ImGui::SliderFloat3("Spin (rad/s)",&g_spinRadSec.x,-200.0f,200.0f);
        ImGui::SliderFloat("Magnus K",&g_magnusK,0.0f,2e-3f,"%.4f");
        ImGui::SliderFloat("Spin decay",&g_spinDecay,0.0f,2.0f);
        ImGui::Checkbox("Drag enabled",&g_dragEnable);
        ImGui::SliderFloat("Drag coeff",&g_dragCoeff,0.0f,1e-3f,"%.5f");

        ImGui::Text("Ball pos: (%.2f, %.2f, %.2f)",
                    g_ballPos.x, g_ballPos.y, g_ballPos.z);
        ImGui::End();

        // camera
        glm::vec3 camPos;
        glm::mat4 view;
        if(g_followCam)
        {
            camPos=g_ballPos+glm::vec3(0.0f,2.0f,10.0f);
            view=glm::lookAt(camPos,g_ballPos,glm::vec3(0,1,0));
        }
        else
        {
            camPos=glm::vec3(0.0f,8.0f,15.0f);
            view=glm::lookAt(camPos,glm::vec3(0.0f,1.0f,0.0f),glm::vec3(0,1,0));
        }

        // render
        glViewport(0,0,wScreen,hScreen);
        glClearColor(0.25f,0.30f,0.35f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);
        glUniformMatrix4fv(uV,1,GL_FALSE,glm::value_ptr(view));
        glUniform3fv(uCP,1,glm::value_ptr(camPos));
        glUniform1f(uR,g_rough);
        glUniform1f(uMet,g_metal);
        glUniform1f(uExp,g_exposure);

        // bind all textures
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D,grassDiff);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D,grassNorm);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D,grassRgh);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D,ballDiff);
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D,ballNorm);
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D,ballRgh);

        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_2D,goalDiff);
        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_2D,goalNorm);
        glActiveTexture(GL_TEXTURE8);
        glBindTexture(GL_TEXTURE_2D,goalRgh);

        glActiveTexture(GL_TEXTURE9);
        glBindTexture(GL_TEXTURE_2D,wallDiff);
        glActiveTexture(GL_TEXTURE10);
        glBindTexture(GL_TEXTURE_2D,wallNorm);
        glActiveTexture(GL_TEXTURE11);
        glBindTexture(GL_TEXTURE_2D,wallRgh);

        // draw grass
        glUniform1i(uType,0);
        {
            glm::mat4 m = glm::mat4(1.0f);
            glUniformMatrix4fv(uM,1,GL_FALSE,glm::value_ptr(m));
            glBindVertexArray(g_planeMesh.vao);
            glDrawElements(GL_TRIANGLES,g_planeMesh.indexCount,GL_UNSIGNED_INT,0);
        }

        // draw wall (4 persons)
        glUniform1i(uType,3);
        {
            float wallZ = -20.0f; 
            float spacing=1.5f;
            float baseX = -1.5f*spacing; // -2.25

            for(int i=0;i<4;++i)
            {
                glm::mat4 mWall=glm::mat4(1.0f);

                float xPos = baseX + i*spacing;
                // place the feet at y=0
                float footOffset = -g_wallVMin.y*g_wallScale; 

                mWall = glm::translate(mWall, glm::vec3(xPos, footOffset, wallZ));
                mWall = glm::scale(mWall, glm::vec3(g_wallScale));

                glUniformMatrix4fv(uM,1,GL_FALSE,glm::value_ptr(mWall));

                glBindVertexArray(wallMesh.vao);
                glDrawElements(GL_TRIANGLES,wallMesh.indexCount,GL_UNSIGNED_INT,nullptr);
            }
        }

        // draw ball
        glUniform1i(uType,1);
        {
            glm::mat4 mBall=glm::mat4(1.0f);
            mBall = glm::translate(mBall,g_ballPos);
            mBall = glm::rotate(mBall,spinAngle,glm::vec3(0.0f,1.0f,0.0f));

            // optional squash & stretch
            static float squash=0.0f;
            float penetration = BALL_RADIUS-g_ballPos.y;
            float impactVel   = -g_v0.y;
            float target=0.0f;
            if(penetration>0.0f && impactVel>0.0f)
                target=glm::clamp(impactVel*0.05f,0.0f,0.5f);
            squash+=(target-squash)*glm::clamp(g_squashRecover*dt,0.0f,1.0f);

            float yScale = 1.0f - squash;
            float xzScale= 1.0f + squash;
            float scaleFactor=3.5f;
            mBall=glm::scale(mBall, glm::vec3(BALL_SCALE*xzScale*scaleFactor,
                                              BALL_SCALE*yScale  *scaleFactor,
                                              BALL_SCALE*xzScale*scaleFactor));
            glUniformMatrix4fv(uM,1,GL_FALSE,glm::value_ptr(mBall));
            glBindVertexArray(ballMesh.vao);
            glDrawElements(GL_TRIANGLES,ballMesh.indexCount,GL_UNSIGNED_INT,0);
        }

        // draw goal
        glUniform1i(uType,2);
        {
            glm::mat4 mGoal=glm::mat4(1.0f);
            mGoal = glm::translate(mGoal, glm::vec3(0.0f,0.0f,g_goalZ));
            mGoal = glm::scale(mGoal, glm::vec3(3.5f));
            glUniformMatrix4fv(uM,1,GL_FALSE,glm::value_ptr(mGoal));
            glBindVertexArray(goalMesh.vao);
            glDrawElements(GL_TRIANGLES,goalMesh.indexCount,GL_UNSIGNED_INT,0);
        }

        // ImGui & swap
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    // cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}
