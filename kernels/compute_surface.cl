#define DEBUG false
#define REJECT true // This mode assigns sentry error value computed error is not less than the given epsilon value
typedef struct {
    float3 focal_point;
    float3 origin;
    float3 u; // This should be the direction vector for the x axis divided by u_mag (so direction of pixel divided by length of pixel squared)
    float3 v; // This should be the direction vector for the y axis, same scaling as u
    float3 N;
    float d;
    int2 res;
} Camera;

typedef struct {
    float3 u; //This on the other hand, should be the length of a pixel in the x direction of screen
    float3 v; //Same for the y direction of the screen
    float3 origin;
} Screen;

typedef struct {
    float3 d; //The search direction, pre-scaled
    int n; //The number of iterations
    float eps;
    float error_value;
} SearchParams;

typedef struct {
    float3 L;
    float3 M;
} Line;


// For the purposes of debugging of the code
void print_point(float3 p) {
    printf("(%f, %f, %f)\n", p.x, p.y, p.z);
}

Line compute_camera_ray(float3 point, __global const Camera *camera) {
    Line camera_ray;
    camera_ray.L = normalize(camera->focal_point - point);
    camera_ray.M = cross(camera_ray.L, point);
    return camera_ray;
}

int2 apply_SGMF(__global const int *SGMF, int2 cam_res, int2 pixel) {
    int2 result;
    result.x = SGMF[pixel.x*cam_res.y + pixel.y];
    result.y = SGMF[(cam_res.x + pixel.x)*cam_res.y + pixel.y];
    return result;
}

int2 compute_camera_pixel(__global const Camera *camera, float3 point, Line camera_ray) {
    float3 pixel_array_pos = (cross(camera_ray.M, camera->N) - camera->d*camera_ray.L)/(dot(camera->N, camera_ray.L)) - camera->origin;
    int2 camera_pixel;
    camera_pixel.x = (int)ceil(dot(pixel_array_pos, camera->u));
    camera_pixel.y = (int)ceil(dot(pixel_array_pos, camera->v));
    return camera_pixel;
}

Line compute_screen_ray(__global const Screen *screen, int2 screen_pixel, float3 point) {
    Line screen_ray; 
    float3 screen_point = screen->origin + screen_pixel.x*screen->u + screen_pixel.y*screen->v;
    screen_ray.L = normalize(screen_point - point);
    screen_ray.M = cross(screen_ray.L, point);
    return screen_ray;
}

float3 compute_normal(Line line1, Line line2){
    float3 normal_vector = line1.L + line2.L;
    float magnitude = length(normal_vector);
    if(magnitude==0) {
        return (float3)(0.0f, 0.0f, 0.0f);
    } else {
        return normal_vector/magnitude;
    }
}

float compute_error(float3 normal_vector_1, float3 normal_vector_2) {
    return 1 - fabs(dot(normal_vector_1, normal_vector_2));
}

inline bool outside_camera_image(__global const Camera* camera, int2 pixel) {
    if(pixel.x < 0 || pixel.y < 0) {
        return true;
    } else if(pixel.x >= camera->res.x || pixel.y >= camera->res.y) {
        return true;
    }
    return false;
}

__kernel void compute_surface(__global const float3 *search_origins,
                            __global const int *SGMF_1,
                            __global const int *SGMF_2,
                            __global float3 *surface,
                            __global const Camera* camera_1,
                            __global const Camera* camera_2,
                            __global const Screen* screen,
                            __global const SearchParams* search_params) {
    int id = get_global_id(0);
    float3 p = search_origins[id];
    float3 point = p;
    point[2] = search_params->error_value;
    float min_err = 1.0f;
    for(int i = 0; i < search_params->n; i++) {
        Line cr_1 = compute_camera_ray(p, camera_1);
        int2 cpx_1 = compute_camera_pixel(camera_1, p, cr_1);
        if (outside_camera_image(camera_1, cpx_1)) {
            p += search_params->d;
            continue;
        }
        Line cr_2 = compute_camera_ray(p, camera_2);
        int2 cpx_2 = compute_camera_pixel(camera_2, p, cr_2);
        if (outside_camera_image(camera_2, cpx_2)) {
            p += search_params->d;
            continue;
        }
        int2 spx_1 = apply_SGMF(SGMF_1, camera_1->res, cpx_1);
        if(spx_1.x == 0 && spx_1.y == 0){
            p += search_params->d;
            continue;
        }
        int2 spx_2 = apply_SGMF(SGMF_2, camera_2->res, cpx_2);
        if(spx_2.x == 0 && spx_2.y == 0){
            p += search_params->d;
            continue;
        }
        Line sr_1 = compute_screen_ray(screen, spx_1, p);
        Line sr_2 = compute_screen_ray(screen, spx_2, p);

        float3 n1 = compute_normal(cr_1, sr_1);
        float3 n2 = compute_normal(cr_2, sr_2);
        float err = compute_error(n1, n2);
        if(DEBUG) {
            printf("Point: ");
            print_point(p);
            printf("Screen Pixel 1: %d, %d\n", spx_1.x, spx_1.y);
            printf("Screen Pixel 2: %d, %d\n", spx_2.x, spx_2.y);
            printf("Normal 1: ");
            print_point(n1);
            printf("Normal 2: ");
            print_point(n2);
            printf("Error: %f\n", err);
            printf("change vector :");
            print_point(search_params->d);
            printf("Updated Point: ");
            print_point(p);
            printf("----------------\n");
        }
        if(err < min_err) { 
            point = p;
            min_err = err;
        }
        p += search_params->d;
    }
    if(!REJECT || min_err < search_params->eps) {
        surface[id] = point;
    } else {
        point[2] = search_params->error_value;
        surface[id] = point;
    }
}

__kernel void test_camera_loading(__global Line* output, __global const float3 *points, __global const Camera* camera) {
    printf("Camera struct size: %d\n", (int)sizeof(Camera));
    printf("Testing Camera DataType\n");
    printf("focal_point = (%f, %f, %f)\n", camera->focal_point.x, camera->focal_point.y, camera->focal_point.z);
    printf("origin = (%f, %f, %f)\n", camera->origin.x, camera->origin.y, camera->origin.z);
    printf("u = (%f, %f, %f)\n", camera->u.x, camera->u.y, camera->u.z);
    printf("v = (%f, %f, %f)\n", camera->v.x, camera->v.y, camera->v.z);
    printf("N = (%f, %f, %f)\n", camera->N.x, camera->N.y, camera->N.z);
    printf("d = %f\n", camera->d);
    printf("res = (%u, %u)\n", camera->res.x, camera->res.y);
}

__kernel void test_screen_loading(__global const Screen* screen) {
    printf("Screen struct size: %d\n", (int)sizeof(Screen));
    printf("Testing Screen DataType\n");
    printf("origin = (%f, %f, %f)\n", screen->origin.x, screen->origin.y, screen->origin.z);
    printf("u = (%f, %f, %f)\n", screen->u.x, screen->u.y, screen->u.z);
    printf("v = (%f, %f, %f)\n", screen->v.x, screen->v.y, screen->v.z);
}

__kernel void test_compute_camera_pixel(__global int2 *output, __global const float3 *points, __global const Camera* camera) {
    int id = get_global_id(0);
    float3 p = points[id];
    printf("starting point = (%f, %f, %f)\n", p.x, p.y, p.z);
    Line cr_1 = compute_camera_ray(p, camera);
    printf("Moment = (%f, %f, %f)\n", cr_1.M.x, cr_1.M.y, cr_1.M.z);
    printf("Direction = (%f, %f, %f)\n", cr_1.L.x, cr_1.L.y, cr_1.L.z);
    int2 cpx_1 = compute_camera_pixel(camera, p, cr_1);
    printf("Pixel = (%d, %d)\n", cpx_1.x, cpx_1.y);
    output[id] = cpx_1;
}

__kernel void test_SGMF_mapping(__global const int *SGMF, __global const int2 *input, __global int2 *output, __global const Screen *screen) {
    int id = get_global_id(0);
    printf("Pixel = (%d, %d)\n", input[0].x, input[0].y);
    printf("Pixel = (%d, %d)\n", input[1].x, input[1].y);
    int2 pixel = input[id];
    int2 cam_res = (int2)(1920, 1080);
    int2 result = apply_SGMF(SGMF, cam_res, pixel);
    printf("Result = (%d, %d)\n", result.x, result.y);
    compute_screen_ray(screen, result, (float3)(0.0f,0.0f,0.0f));
    output[id] = result;
}
