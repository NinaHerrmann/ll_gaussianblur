/**
 * Copyright (c) 2020 Nina Herrmann
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */
#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
//#include <mpi.h>
#define MAX_ITER 1000
#ifdef __CUDACC__
#define POW(a, b)      powf(a, b)
#define EXP(a)      exp(a)
#else
#define POW(a, b)      std::pow(a, b)
#define EXP(a)      std::exp(a)
#endif
int rows, cols;
int* input_image_int;
char* input_image_char;
bool ascii = false;
int DEFAULT_TILE_WIDTH = 16;
bool DEBUG = false;
int stencil_size = 2;
int block_mult = 1;
#define gpuErrchk(ans)                                                         \
{ gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

int writePGM(const std::string& filename, int *out_image, int rows, int cols, int max_color)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cout << "Error: Cannot open image file " << filename << "!" << std::endl;
        return 1;
    }

    // Gather full image
    int** img = new int*[rows];
    for (int i = 0; i < rows; i++)
        img[i] = new int[cols];

    // Write image header
    ofs << "P5\n" << cols << " " << rows << " " << std::endl << max_color << std::endl;

    // Write image
    for (int x = 0; x < rows; x++) {
        for (int y = 0; y < cols; y++) {
            unsigned char intensity = static_cast<unsigned char> (out_image[x*cols + y]);
            ofs << intensity;
        }
    }
    if (ofs.fail()) {
        std::cout << "Cannot write file " << filename << "!" << std::endl;
        return 1;
    }

    return 0;
}
int readPGM(const std::string& filename, int& rows, int& cols, int& max_color)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        std::cout << "Error: Cannot open image file " << filename << "!" << std::endl;
        return 1;
    }
    // Read magic number.
    std::string magic;
    getline(ifs, magic);
    if (magic.compare("P5")) { // P5 is magic number for pgm binary format.
        if (magic.compare("P2")) { // P2 is magic number for pgm ascii format.
            std::cout << "Error: Image not in PGM format!" << std::endl;
            return 1;
        }
        ascii = true;
    }

    // Skip comments
    std::string inputLine;
    while (true) {
        getline(ifs, inputLine);
        if (inputLine[0] != '#') break;
    }

    // Read image size and max color.
    std::stringstream(inputLine) >> cols >> rows;
    getline(ifs, inputLine);
    std::stringstream(inputLine) >> max_color;
   // std::cout << "\nmax_color: " << max_color << "\t cols: " << cols << "\t rows: " << rows << std::endl;

    // Read image.
    if (ascii) {
        input_image_int = new int[rows*cols];
        int i = 0;
        while (getline(ifs, inputLine)) {
            std::stringstream(inputLine) >> input_image_int[i++];
        }
    } else {
        input_image_char = new char[rows*cols];
        ifs.read(input_image_char, rows*cols);
    }
    int * gs_image = (int*)malloc(rows*cols*sizeof(int));
    for (int x = 0; x < rows*cols; x++) {
        gs_image[x] = input_image_char[x] + '0';
    }

    const std::string in_file = "sfirstout";
    writePGM(in_file, gs_image, rows, cols, max_color);


    return 0;
}

__device__
void printsm(int global_col, int global_row, int tile_width, int * data){
    if (global_col == 0 && global_row == 0) {
        printf("[");
        for(int i = 0; i < tile_width; i++){
            printf("\n");
            for(int j = 0; j < tile_width; j++){
                printf("%d;", data[j + i * tile_width]);
            }
        }
        printf("]\n");
    }
}
__device__
void printgm(int global_col, int global_row, int tile_width, int cols, int * data){
    if (global_col == 0 && global_row == 0) {
        printf("[");
        for(int i = 0; i < tile_width; i++){
            printf("\n");
            for(int j = 0; j < tile_width; j++){
                printf("%d;", data[j + i * cols]);
            }
        }
        printf("]\n");
    }
}

__global__
void calcGaussian(const int *input, int *output, int cols, int kw) {
    size_t thread = threadIdx.x + blockIdx.x * blockDim.x;

    int row = thread / cols;
    int col = thread % cols;
    int offset = kw/2;
    float weight = 0.5f;
    float sigma = 1;
    float mean = (float)kw/2;
    //printgm(row, col, 32, cols+kw, input);
    float sum = 0;
    for (int r = 0; r <= kw; ++r) {
        for (int c = 0; c <= kw; ++c) {
            sum += input[(row + r) * (cols + kw) + col + c] *
                    EXP(-0.5 * (POW((r-mean)/sigma, 2.0) + POW((c-mean)/sigma,2.0))) / (2 * M_PI * sigma * sigma);
        }
    }
    output[(row * (cols+kw) + (offset * (cols+kw))) + (col + offset)] = (int)sum/weight;

}

__global__
void calcGaussianSM(const int *input, int *output, int cols, int rows, int kw, int tile_width) {

    int global_col = blockIdx.y * blockDim.y + threadIdx.y;
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    int local_col = threadIdx.y;
    int local_row = threadIdx.x;
    const int offset = kw/2;
    int writeto = (global_col + offset) + global_row * (cols+kw) + (offset * (cols+kw));
    const float weight = 1.0f;
    const float sigma = 1;
    const int new_tile_width = tile_width + kw;
    const int inside_elements = tile_width * tile_width;
    const int iterations = ((new_tile_width * new_tile_width) / (inside_elements)) + 1;
    extern __shared__ int data[];
    // Fills "sequentially"
    for (int r = 0; r <= iterations; ++r) {
        int local_index = (r * (inside_elements)) + (local_row) * tile_width + (local_col);
        int row = local_index / new_tile_width;
        int firstcol = global_col - local_col;
        int g_col = firstcol + ((local_index) % new_tile_width);
        int readfrom = (((global_row-local_row) + row) * (cols+kw)) + g_col;
        if (local_index <= new_tile_width * new_tile_width) {
            data[local_index] = input[readfrom];
        }
    }

    __syncthreads();
    //printsm(global_row, global_col, new_tile_width, data);
    //
    double sum0 = 0;
    const double divisor = (2 * M_PI * sigma * sigma);
    for (int r = 0; r < kw; ++r) {
        for (int c = 0; c < kw; ++c) {
           sum0 += data[(local_row + r) * new_tile_width + (local_col + c)] *
                   EXP(-0.5 * (POW((r-offset)/sigma, 2.0) + POW((c-offset)/sigma,2.0))) / divisor;
        }
    }
    output[writeto] = (int)(sum0/weight);
}
__global__
void calcGaussianSM_double(const int *input, int *output, int cols, int rows, int kw, int tile_width, int block_mult) {

    int global_col = blockIdx.y * blockDim.y + threadIdx.y;
    int global_row = (blockIdx.x * blockDim.x + threadIdx.x) +
            (tile_width * ((blockIdx.x * blockDim.x + threadIdx.x) / tile_width));
    int local_col = threadIdx.y;
    int local_row = threadIdx.x;
    const int offset = kw/2;
    int writeto = (global_col + offset) + global_row * (cols+kw) + (offset * (cols+kw));
    const float weight = 1.0f;
    const float sigma = 1;
    const int new_tile_width = tile_width + kw;
    const int inside_elements = tile_width * tile_width;
    const int newsize = new_tile_width * new_tile_width + (new_tile_width * (tile_width));
    const int iterations = block_mult * ((new_tile_width * new_tile_width) / (inside_elements)) + 1;
    extern __shared__ int data[];
    // Fills "sequentially"
    // load "double" the elements to SM

    for (int r = 0; r <= iterations; ++r) {
        int local_index = (r * (inside_elements)) + (local_row) * tile_width + (local_col);
        int row = local_index / new_tile_width;
        int firstcol = global_col - local_col;
        int g_col = firstcol + ((local_index) % new_tile_width);
        int readfrom = (((global_row-local_row) + row) * (cols+kw)) + g_col;
        if (local_index <= newsize) {
            data[local_index] = input[readfrom];
        }
    }
    __syncthreads();
    //printsm(global_row, global_col, new_tile_width, data);
    //
    double sum0 = 0;
    double sum1 = 0;
    const double divisor = (2 * M_PI * sigma * sigma);
    for (int r = 0; r < kw; ++r) {
        for (int c = 0; c < kw; ++c) {
            //if ((local_row + r) * new_tile_width + (local_col + c) >= (new_tile_width * new_tile_width)){ printf("uupss\n");}
            int local_read_index = (local_row + r) * new_tile_width + (local_col + c);
            sum0 += data[local_read_index] *
                    EXP(-0.5 * (POW((r-offset)/sigma, 2.0) + POW((c-offset)/sigma,2.0))) / divisor;
            sum1 += data[local_read_index + (new_tile_width * (tile_width + offset))] *
                    EXP(-0.5 * (POW((r-offset)/sigma, 2.0) + POW((c-offset)/sigma,2.0))) / divisor;
        }
    }
    output[writeto] = (int)(sum0/weight);
    output[writeto + ((tile_width) * (cols+kw))] = (int)(sum1/weight);
}
__global__
void old_calcGaussianSM(const int *input, int *output, int cols, int rows, int kw, int tile_width) {

    int global_col = blockIdx.y * blockDim.y + threadIdx.y;
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    int local_col = threadIdx.y;
    int local_row = threadIdx.x;
    int offset = kw/2;
    int writeto = (global_col + offset) + global_row * (cols+kw) + (offset * (cols+kw));
    float weight = 1.0f;
    float sigma = 1;
    int new_tile_width =  tile_width + kw;
    int inside_elements = tile_width * tile_width;
    float mean = (float)kw/2;
    extern __shared__ int data[];


    for (int r = 0; r <= kw; ++r) {
         for (int c = 0; c <= kw; ++c) {
             data[(local_row + r) * new_tile_width + (local_col + c)] =
             input[((global_row + r) * (cols+kw)) + (global_col + c)];
         }
     }
    __syncthreads();
    //printsm(global_row, global_col, new_tile_width, data);
    double sum = 0;
    double divisor = (2 * M_PI * sigma * sigma);
    for (int r = 0; r <= kw; ++r) {
        for (int c = 0; c <= kw; ++c) {
            sum += data[(local_row + r) * new_tile_width + (local_col + c)] *
                    EXP(-0.5 * (POW((r-mean)/sigma, 2.0) + POW((c-mean)/sigma,2.0))) / divisor;
        }
    }
    output[writeto] = (int)(sum/weight);
}
__global__
void calcGaussian_fixed_SM(const int *input, int *output, int cols, int rows, int kw, int tile_width) {

    int global_col = blockIdx.y * blockDim.y + threadIdx.y;
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    int local_col = threadIdx.y;
    int local_row = threadIdx.x;
    int offset = kw/2;
    float weight = 1.0f;
    float sigma = 1;
    int new_tile_width =  tile_width + kw;
    float mean = (float)kw/2;
    __shared__ int data[1764];
    for (int r = 0; r <= kw; ++r) {
        for (int c = 0; c <= kw; ++c) {
            data[(local_row + r) * new_tile_width + (local_col + c)] = input[((global_row + r) * (cols+kw)) + (global_col + c)];
        }
    }
    __syncthreads();
    //printsm(global_row, global_col, new_tile_width, data);
    float sum = 0;
    for (int r = 0; r <= kw; ++r) {
        for (int c = 0; c <= kw; ++c) {
            sum += data[(local_row + r) * new_tile_width + (local_col + c)] *
                    EXP(-0.5 * (POW((r-mean)/sigma, 2.0) + POW((c-mean)/sigma,2.0))) / (2 * M_PI * sigma * sigma);
        }
    }
    output[(global_col + offset) + global_row * (cols+kw) + (offset * (cols+kw))] = (int)sum/weight;
}
float testGaussian(std::string in_file, std::string out_file, bool output, int tile_width, int iterations, int iterations_used, std::string file, bool shared_mem, int kw) {
    int max_color;
    cudaEvent_t initstart, initstop;
    cudaEventCreate(&initstart);
    cudaEventCreate(&initstop);
    cudaEventRecord(initstart);

    // Read image
    readPGM(in_file, rows, cols, max_color);
    const unsigned int elements = (rows + kw) * (cols + kw);
    const unsigned int stencilmatrixsize = elements * sizeof(int);
    int * gs_image = (int*)malloc(stencilmatrixsize);
    memset(gs_image, 0, stencilmatrixsize);

    if (ascii) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int offset = tile_width + kw + kw/2 + (i*(cols+kw));
                gs_image[i + offset] = input_image_int[i];
            }
        }
    } else {
        int stencil = kw/2;
        for (int i = 0; i < rows+kw; i++) {
            for (int j = 0; j < cols+kw; j++) {
                if (i < (kw/2) || i > (rows+stencil) || j < (kw/2) || j > (cols+stencil)){
                    gs_image[(i*(rows+kw)) + j] = 0;
                } else {
                    int offset = ((i-stencil)*cols) + (j-stencil);
                    gs_image[(i*(cols+kw)) + j] = input_image_char[offset] + '0';
                }
            }
        }
    }
    // 5316480 - 7962624
    int *d_gs_image;
    int *d_gs_image_result;
    cudaMalloc((int**)&d_gs_image, stencilmatrixsize);
    cudaMalloc((int**)&d_gs_image_result, stencilmatrixsize);
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaMemcpy(d_gs_image, gs_image, stencilmatrixsize, cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    int smem_size = (tile_width + kw) * (tile_width + kw) * sizeof(int) * 2;
    cudaEventRecord(initstop);
    cudaEventSynchronize(initstop);
    float initmilliseconds = 0;
    cudaEventElapsedTime(&initmilliseconds, initstart, initstop);
    if (true) {
        if (output) {
            std::ofstream outputFile;
            outputFile.open(file, std::ios_base::app);
            outputFile << "" << initmilliseconds/1000 << ";";
            //printf("%.2f", initmilliseconds/1000);
            outputFile.close();
        }
    }
 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int run = 0; run < iterations; ++run) {
        // TODO make multiple GPUs
        int threads = 128;
        if(!shared_mem){
            dim3 dimBlock(128);
            dim3 dimGrid(((rows*cols)) / dimBlock.x);
            calcGaussian<<<dimGrid, dimBlock, smem_size, stream1>>>(d_gs_image, d_gs_image_result, cols, kw);
            calcGaussian<<<dimGrid, dimBlock, smem_size, stream1>>>(d_gs_image_result, d_gs_image, cols, kw);

        } else{
                dim3 dimBlock(tile_width, tile_width);
                dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x / block_mult,
                             (cols + dimBlock.y - 1) / dimBlock.y);
                calcGaussianSM_double<<<dimGrid, dimBlock, smem_size, stream1>>>(d_gs_image, d_gs_image_result, cols,
                                                                                 rows, kw, tile_width, block_mult);
                calcGaussianSM_double<<<dimGrid, dimBlock, smem_size, stream1>>>(d_gs_image_result, d_gs_image, cols,
                                                                                 rows, kw, tile_width, block_mult);
        }
        if (DEBUG) {
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    if (true) {
            std::ofstream outputFile;
            outputFile.open(file, std::ios_base::app);
            //outputFile << "" << milliseconds/1000 << ";";
            std::cout << "" << milliseconds/1000 << ";";

            outputFile.close();

    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    writePGM(out_file, gs_image, rows+kw, cols+kw, max_color);

    cudaEventRecord(start);

    for (int run = 0; run < iterations; ++run) {
        // TODO make multiple GPUs
        int threads = 128;
        if(!shared_mem){
            dim3 dimBlock(128);
            dim3 dimGrid(((rows*cols)) / dimBlock.x);
            calcGaussian<<<dimGrid, dimBlock, smem_size, stream1>>>(d_gs_image, d_gs_image_result, cols, kw);
            calcGaussian<<<dimGrid, dimBlock, smem_size, stream1>>>(d_gs_image_result, d_gs_image, cols, kw);

        } else{
            dim3 dimBlock(tile_width, tile_width);
            dim3 dimGrid((rows + dimBlock.x - 1) / dimBlock.x,
                         (cols + dimBlock.y - 1) / dimBlock.y);
            calcGaussianSM<<<dimGrid, dimBlock, smem_size, stream1>>>(d_gs_image, d_gs_image_result, cols,
                                                                             rows, kw, tile_width);
            calcGaussianSM<<<dimGrid, dimBlock, smem_size, stream1>>>(d_gs_image_result, d_gs_image, cols,
                                                                             rows, kw, tile_width);
        }
        if (DEBUG) {
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    if (true) {
            std::ofstream outputFile;
            outputFile.open(file, std::ios_base::app);
            //outputFile << "" << milliseconds/1000 << ";";
            std::cout << "" << milliseconds/1000 << ";";

            outputFile.close();

    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // TODO COPY BACK
    cudaMemcpy(gs_image, d_gs_image_result, stencilmatrixsize, cudaMemcpyDeviceToHost);

    return milliseconds;
}

int init(int row, int col)
{
    if (ascii) return input_image_int[row*cols+col];
    else return input_image_char[row*cols+col];
}
int main(int argc, char **argv) {
    //std::cout << "\n\n************* Starting the Gaussian Blur *************\n ";

    int nGPUs = 1;
    int nRuns = 1;
    int iterations = MAX_ITER;
    int tile_width = DEFAULT_TILE_WIDTH;
    float cpu_fraction = 0.0;
    //bool warmup = false;
    bool output = false;
    bool shared_mem = false;
    int kw = 2;

    std::string in_file, out_file, file, nextfile; //int kw = 10;
    file = "result_travel.csv";
    if (argc >= 9) {
        nGPUs = atoi(argv[1]);
        nRuns = atoi(argv[2]);
        cpu_fraction = atof(argv[3]);
        if (cpu_fraction > 1) {
            cpu_fraction = 1;
        }
        tile_width = atoi(argv[4]);
        iterations = atoi(argv[5]);
        if (atoi(argv[6]) == 1) {
            shared_mem = true;
        }
        kw = atoi(argv[7]);
        block_mult = atoi(argv[8]);

    }
    printf("%d;", block_mult);
    std::string shared = shared_mem ? "SM" : "GM";

    if (argc == 10) {
        in_file = argv[9];
        size_t pos = in_file.find(".");
        out_file = in_file;
        std::stringstream ss;
        ss << "_" << nGPUs << "_" << iterations << "_" << shared <<  "_" << tile_width << "_" << kw << "_gaussian";
        out_file.insert(pos, ss.str());
    } else {
        in_file = "ungaro4k.pgm";
        std::stringstream oo;
        oo << in_file << "_" << block_mult << "_" << nGPUs << "_" << iterations << "_" << shared <<  "_" << tile_width << "_" << kw << "_gaussian.pgm";
        out_file = oo.str();
    }
    output = true;
    std::stringstream ss;
    ss << file << "_" << iterations;
    nextfile = ss.str();

    int iterations_used = 0;
    float overalltime = 0.0;
    for (int r = 0; r < nRuns; ++r) {
        overalltime += testGaussian(in_file, out_file, output, tile_width, iterations, iterations_used, nextfile, shared_mem, kw);
    }

    if (output) {
        std::ofstream outputFile;
        outputFile.open(nextfile, std::ios_base::app);
        outputFile << "" + std::to_string(nGPUs) + ";"  + std::to_string(block_mult) + ";"
        + std::to_string(tile_width) + ";" + std::to_string(iterations) + ";" +
        std::to_string(iterations_used) + ";" + std::to_string(overalltime/nRuns) + ";\n";
        outputFile.close();
    }
    //std::cout << "\n************* Finished the Gaussian Blur *************\n ";
    std::cout << "" + std::to_string(nGPUs) + ";" + std::to_string(tile_width) + ";\n";

    return 0;
}
