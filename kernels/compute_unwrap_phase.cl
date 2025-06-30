__kernel void  generate_data(__global const uchar *images,
                             __global float *output,
                             const int width,
                             const int height,
                             const int N) {
    int image_id = get_global_id(0);
    int x = get_global_id(1);
    int y = get_global_id(2);

    int camera = image_id / (2*N);
    int axis = (image_id / N) % 2;
    int wavenumber = image_id % N;

    int output_index = (((camera*2+axis)*N+wavenumber)*width+x)*height+y;

    int image_size = width*height;
    
    float I_0 = (float)images[image_id*image_size*4 + y*width + x];
    float I_1 = (float)images[image_id*image_size*4 + image_size + y*width + x];
    float I_2 = (float)images[image_id*image_size*4 + image_size*2 + y*width + x];
    float I_3 = (float)images[image_id*image_size*4 + image_size*3 + y*width + x];
    float num = I_3 - I_1;
    float den = I_0 - I_2;

    if(den==0) {
        output[output_index] = 0;
    } else {
        output[output_index] = atan(num/den) + 1.57079632679f;
    }
}
