// ---------------------------------------------------------------------------
// main.cpp  ―  Soccer-ball free-flight demo with grass field and ImGui camera
// ---------------------------------------------------------------------------

#include <iostream>
#include <cmath>
#define _USE_MATH_DEFINES            // ensure M_PI on MSVC

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

// stb_image  (single-file header for loading textures)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ---------------------------------------------------------------------------
// GLSL shaders (soccer-ball OR grass, decided by uniform “isGrass”)
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

uniform vec3  lightPos, camPos, lightColor;
uniform float roughness, metallic;
uniform bool  isGrass;

uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D roughnessMap;

// --- Soccer-ball colours ---------------------------------------------------
const vec3 WHITE = vec3(1.5);   // >1 for extra bright white
const vec3 BLACK = vec3(0.05);

// 12 patch centres in UV for a quick “soccer” look
vec3 soccerColor(vec2 uv)
{
    vec2 C[12] = vec2[](
        vec2(0.50,0.50), vec2(0.25,0.70), vec2(0.75,0.70),
        vec2(0.25,0.30), vec2(0.75,0.30), vec2(0.10,0.50),
        vec2(0.90,0.50), vec2(0.50,0.10), vec2(0.50,0.90),
        vec2(0.15,0.15), vec2(0.85,0.85), vec2(0.15,0.85)
    );
    for(int i=0;i<12;i++)
        if(distance(uv,C[i])<0.15) return BLACK;
    return WHITE;
}

// --- Minimal Cook–Torrance (GGX) helpers -----------------------------------
float D_GGX(vec3 N, vec3 H, float a)
{
    float a2=a*a, NdotH=max(dot(N,H),0.0);
    float denom=(NdotH*NdotH*(a2-1.0)+1.0);
    return a2/(3.14159*denom*denom);
}
float G_Schlick(float NdotV,float k){return NdotV/(NdotV*(1.0-k)+k);}
float G_Smith(vec3 N,vec3 V,vec3 L,float a)
{
    float k=(a+1.0); k=k*k/8.0;
    return G_Schlick(max(dot(N,V),0.0),k)*G_Schlick(max(dot(N,L),0.0),k);
}
vec3  F_Schlick(float cosT,vec3 F0){return F0+(1.0-F0)*pow(1.0-cosT,5.0);}

void main()
{
    // Base colour / material from either grass textures or soccer pattern
    vec3  N  = normalize(vNorm);
    vec3  V  = normalize(camPos - vPos);
    vec3  L  = normalize(lightPos - vPos);
    vec3  H  = normalize(V+L);

    vec3  baseCol;
    float a = roughness;   // surface roughness
    float m = metallic;    // metallic 0-1

    if(isGrass)
    {
        vec2 uv = vUv * 50.0;                             // tile grass
        baseCol = texture(albedoMap, uv).rgb;
        N       = normalize(texture(normalMap, uv).rgb*2.0-1.0);
        a       = texture(roughnessMap, uv).r;
        m       = 0.0;
    }
    else
    {
        baseCol = soccerColor(vUv);
    }

    // Cook–Torrance
    vec3  F0 = mix(vec3(0.04), baseCol, m);
    vec3  F  = F_Schlick(max(dot(H,V),0.0), F0);
    float D  = D_GGX(N,H,a);
    float G  = G_Smith(N,V,L,a);

    vec3  spec = (D*G*F) /
                 (4.0*max(dot(N,V),0.0)*max(dot(N,L),0.0)+0.001);
    vec3  diff = (1.0-F)* (1.0-m) * baseCol / 3.14159;

    vec3  colour = (diff+spec)*lightColor*max(dot(N,L),0.0);

    FragColor = vec4(colour,1.0);
}
)";

// ---------------------------------------------------------------------------
// Geometry builders (sphere + plane)
// ---------------------------------------------------------------------------
void buildSphere(int sectors,int stacks,
                 std::vector<float>& vb,std::vector<unsigned>& ib)
{
    for(int i=0;i<=stacks;++i){
        float phi=M_PI/2-i*(M_PI/stacks);
        float y=sin(phi), r=cos(phi);
        for(int j=0;j<=sectors;++j){
            float theta=j*(2*M_PI/sectors);
            float x=r*cos(theta), z=r*sin(theta);
            vb.insert(vb.end(),{x,y,z,x,y,z,
                                (float)j/sectors,(float)i/stacks});
        }
    }
    for(int i=0;i<stacks;++i)
        for(int j=0;j<sectors;++j){
            int a=i*(sectors+1)+j, b=a+sectors+1;
            ib.insert(ib.end(),{(unsigned)a, (unsigned)b, (unsigned)(a+1), (unsigned)b, (unsigned)(b+1), (unsigned)(a+1)});
        }
}
void buildPlane(std::vector<float>& vb,std::vector<unsigned>& ib,float size=100.f)
{
    float v[]={
        -size,0,-size, 0,1,0, 0,0,
         size,0,-size, 0,1,0, 1,0,
         size,0, size, 0,1,0, 1,1,
        -size,0, size, 0,1,0, 0,1};
    unsigned idx[]={0,1,2,2,3,0};
    vb.insert(vb.end(),std::begin(v),std::end(v));
    ib.insert(ib.end(),std::begin(idx),std::end(idx));
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------
GLuint compile(GLenum type,const char* src)
{
    GLuint s=glCreateShader(type);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){char log[512]; glGetShaderInfoLog(s,512,nullptr,log);
        std::cerr<<"Shader error:\n"<<log<<'\n'; }
    return s;
}
GLuint loadTex(const char* path)
{
    int w,h,c; stbi_set_flip_vertically_on_load(true);
    unsigned char* data=stbi_load(path,&w,&h,&c,0);
    if(!data){ std::cerr<<"Texture load fail: "<<path<<'\n'; return 0;}
    GLenum fmt=(c==1)?GL_RED:(c==3)?GL_RGB:GL_RGBA;
    GLuint id; glGenTextures(1,&id); glBindTexture(GL_TEXTURE_2D,id);
    glTexImage2D(GL_TEXTURE_2D,0,fmt,w,h,0,fmt,GL_UNSIGNED_BYTE,data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    stbi_image_free(data); return id;
}

// ---------------------------------------------------------------------------
// Global simulation parameters
// ---------------------------------------------------------------------------
float  g_gravity = -9.81f;
glm::vec3 g_ballPos(0);
glm::vec3 g_v0(10.f,8.f,0.f);
float g_tAccum = 0.f;
bool  g_followCam = false;
float g_rough = 0.3f, g_metal = 0.0f;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int main()
{
    // ---------- GLFW / OpenGL context --------------------------------------
    if(!glfwInit()){ std::cerr<<"GLFW init fail\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    GLFWwindow* win = glfwCreateWindow(1280,720,"Soccer Demo",nullptr,nullptr);
    if(!win){ glfwTerminate(); return -1;}
    glfwMakeContextCurrent(win);
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cerr<<"GLAD init fail\n"; return -1;}
    glfwSwapInterval(1);                               // vsync

    // ---------- ImGui ------------------------------------------------------
    IMGUI_CHECKVERSION(); ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(win,true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // ---------- Shader program --------------------------------------------
    GLuint vs=compile(GL_VERTEX_SHADER,VERT_SHADER);
    GLuint fs=compile(GL_FRAGMENT_SHADER,FRAG_SHADER);
    GLuint prog=glCreateProgram();
    glAttachShader(prog,vs); glAttachShader(prog,fs); glLinkProgram(prog);

    // ---------- Geometry (sphere & plane) ----------------------------------
    std::vector<float> vbSphere; std::vector<unsigned> ibSphere;
    buildSphere(64,64,vbSphere,ibSphere);
    GLuint vaoS,vboS,eboS; glGenVertexArrays(1,&vaoS);
    glGenBuffers(1,&vboS); glGenBuffers(1,&eboS);
    glBindVertexArray(vaoS);
    glBindBuffer(GL_ARRAY_BUFFER,vboS);
    glBufferData(GL_ARRAY_BUFFER,vbSphere.size()*4,vbSphere.data(),GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,eboS);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,ibSphere.size()*4,ibSphere.data(),GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,8*4,(void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,8*4,(void*)(3*4)); glEnableVertexAttribArray(1);
    glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,8*4,(void*)(6*4)); glEnableVertexAttribArray(2);
    glBindVertexArray(0);

    std::vector<float> vbPlane; std::vector<unsigned> ibPlane;
    buildPlane(vbPlane,ibPlane);
    GLuint vaoP,vboP,eboP; glGenVertexArrays(1,&vaoP);
    glGenBuffers(1,&vboP); glGenBuffers(1,&eboP);
    glBindVertexArray(vaoP);
    glBindBuffer(GL_ARRAY_BUFFER,vboP);
    glBufferData(GL_ARRAY_BUFFER,vbPlane.size()*4,vbPlane.data(),GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,eboP);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,ibPlane.size()*4,ibPlane.data(),GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,8*4,(void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,8*4,(void*)(3*4)); glEnableVertexAttribArray(1);
    glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,8*4,(void*)(6*4)); glEnableVertexAttribArray(2);
    glBindVertexArray(0);

    // ---------- Load textures ---------------------------------------------
    GLuint texDiff = loadTex("grass_diffuse.png");
    GLuint texNorm = loadTex("grass_normal.png");
    GLuint texRgh  = loadTex("grass_roughness.png");

    // ---------- Uniform locations -----------------------------------------
    glUseProgram(prog);
    GLint u_m  = glGetUniformLocation(prog,"model");
    GLint u_v  = glGetUniformLocation(prog,"view");
    GLint u_p  = glGetUniformLocation(prog,"proj");
    GLint u_lp = glGetUniformLocation(prog,"lightPos");
    GLint u_lc = glGetUniformLocation(prog,"lightColor");
    GLint u_cp = glGetUniformLocation(prog,"camPos");
    GLint u_r  = glGetUniformLocation(prog,"roughness");
    GLint u_met= glGetUniformLocation(prog,"metallic");
    GLint u_is = glGetUniformLocation(prog,"isGrass");
    glUniform3f(u_lp,0.f,30.f,30.f);
    
    glUniform3f(u_lc,3.f,3.f,3.f);
    static float exposure = 1.5f;
    // texture samplers
    glUniform1i(glGetUniformLocation(prog,"albedoMap"),0);
    glUniform1i(glGetUniformLocation(prog,"normalMap"),1);
    glUniform1i(glGetUniformLocation(prog,"roughnessMap"),2);

    // projection matrix (fixed)
    int winW,winH; glfwGetFramebufferSize(win,&winW,&winH);
    glm::mat4 proj = glm::perspective(glm::radians(45.f),
                                      float(winW)/winH,0.1f,200.f);
    glUniformMatrix4fv(u_p,1,GL_FALSE,glm::value_ptr(proj));

    // GL states
    glEnable(GL_DEPTH_TEST);

    // ---------- Timing -----------------------------------------------------
    double lastT = glfwGetTime();
    float  ballSpin = 0.f;

    // ---------- Main loop --------------------------------------------------
    while(!glfwWindowShouldClose(win))
    {
        // -- time step --
        double now = glfwGetTime(); float dt=float(now-lastT); lastT=now;

        glfwPollEvents();

        // -- Physics update: simple parabola --
        g_tAccum += dt;
        g_ballPos.x = g_v0.x * g_tAccum;
        g_ballPos.y = g_v0.y * g_tAccum + 0.5f*g_gravity*g_tAccum*g_tAccum;
        g_ballPos.z = g_v0.z * g_tAccum;
        if (g_ballPos.y < 0.f) {         // 地面在 y = 0
            g_tAccum = 0.f;
            g_ballPos.y = 0.f;           // 只把高度拉回地面
            // 如果想要反弹而不是停住：
            // g_v0.y *= -0.6f;          // 反弹系数
        }
        ballSpin += 1.f*dt;

        // -------------- ImGui frame ---------------------------------------
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Controls");
        ImGui::Checkbox("Follow Ball Camera",&g_followCam);
        if(ImGui::Button("Reset Ball")){
            g_tAccum=0.f; g_ballPos=glm::vec3(0);
        }
        ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f);
        ImGui::SliderFloat3("Initial Velocity", &g_v0.x, -30.f, 30.f);
        ImGui::SliderFloat3("Initial Velocity",&g_v0.x,-20.f,20.f);
        ImGui::SliderFloat("Roughness",&g_rough,0.f,1.f);
        ImGui::SliderFloat("Metallic",&g_metal,0.f,1.f);
        ImGui::Text("Ball (x,y,z): %.2f %.2f %.2f",
                    g_ballPos.x,g_ballPos.y,g_ballPos.z);
        ImGui::End();

        // -------------- Camera logic --------------------------------------
        static bool lastFollow = false;
        if(g_followCam!=lastFollow){
            std::cout<<"[Camera] Mode: "
                     <<(g_followCam?"Follow Ball":"Fixed")<<'\n';
            lastFollow=g_followCam;
        }
        glm::vec3 camPos;
        glm::mat4 view;
        if(g_followCam){
            camPos = g_ballPos + glm::vec3(0,2,8);
            view   = glm::lookAt(camPos,g_ballPos,glm::vec3(0,1,0));
        }else{
            camPos = glm::vec3(-10, 5, 20);                   
            view   = glm::lookAt(camPos,
                                glm::vec3(0, 1, 0),         
                                glm::vec3(0, 1, 0));
        }

        // -------------- Render pass ---------------------------------------
        glClearColor(0.2f,0.3f,0.35f,1.f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);
        glUniformMatrix4fv(u_v,1,GL_FALSE,glm::value_ptr(view));
        glUniform3f(u_cp,camPos.x,camPos.y,camPos.z);
        glUniform1f(u_r,g_rough);
        glUniform1f(u_met,g_metal);

        // bind grass textures
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,texDiff);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D,texNorm);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D,texRgh);

        // -- draw plane (grass) --
        glUniform1i(u_is,1);
        glm::mat4 modelPlane(1.f);
        glUniformMatrix4fv(u_m,1,GL_FALSE,glm::value_ptr(modelPlane));
        glBindVertexArray(vaoP);
        glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_INT,0);

        // -- draw soccer ball --
        glUniform1i(u_is,0);
        glm::mat4 modelBall(1.f);
        modelBall = glm::translate(modelBall,g_ballPos);
        modelBall = glm::scale(modelBall,glm::vec3(1.0f));
        modelBall = glm::rotate(modelBall,ballSpin,glm::vec3(0,1,0));
        glUniformMatrix4fv(u_m,1,GL_FALSE,glm::value_ptr(modelBall));
        glBindVertexArray(vaoS);
        glDrawElements(GL_TRIANGLES,(GLsizei)ibSphere.size(),
                       GL_UNSIGNED_INT,0);

        // -------------- ImGui render --------------------------------------
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    // ---------- Cleanup ----------------------------------------------------
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}
