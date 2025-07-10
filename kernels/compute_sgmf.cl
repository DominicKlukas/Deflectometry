#define M_2PI 6.2831853071795864f
#define SGMF_ERROR_VALUE -1000
__kernel void  generate_data(__global float *data,
                         __global int *sgmf,
                         __const int width,
                         __const int height,
                         __const float eps,
                         __const int max_N, // This is the max that will be used in the computation
                         __const int N, // This is the number of wavenumbers in the dataset
                         __const int scr_res_x,
                         __const int scr_res_y
                         ) {
    // These comes from the global_size tuple, and enumerate all of the work items
    // There are 2 SGMF's for the two cameras, each with two dimensions
    int id = get_global_id(0); int x = get_global_id(1); // X pixel index
    int y = get_global_id(2); // Y pixel index

    int camera = id / 2;
    int ax = id % 2;

    int sgmf_idx = ((camera*2 + ax)*width + x)*height + y;
    int data_idx = ((camera*2 + ax)*N*width + x) * height + y;

    int k = 1;
    float r = data[data_idx];  // wavenumber 0

    for (int wn = 1; wn < max_N; wn++) {
        k *= 2;
        data_idx += width*height; //This will advance the image to the image with the next wavenumber, while keeping the same pixel index
        float next = data[data_idx];
        float phase_dif = fabs(((1.0f / k) * next - r) / M_PI_2);
        float n = floor(phase_dif + eps);
        r = next + n * M_PI;
    }

    // Final scaling: r * scr_res[ax] / (2π × 2^(max_N - 2))
    if (ax == 0) {
        int value = (int)(r*(float)scr_res_x / (M_2PI * (1 << (max_N - 2))));
        if(value >= 0 && value < scr_res_x){
            sgmf[sgmf_idx] = value;
        } else {
            sgmf[sgmf_idx] = SGMF_ERROR_VALUE;
        }
    }
    else {
        int value = (int)(r*(float)scr_res_y / (M_2PI * (1 << (max_N - 2))));
        if(value >= 0 && value < scr_res_y){
            sgmf[sgmf_idx] = value;
        } else {
            sgmf[sgmf_idx] = SGMF_ERROR_VALUE;
        }
    }
}
