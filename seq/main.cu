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
//#include <cmath>
#include <mpi.h>
#define MAX_ITER 1000
int rows, cols;
int* input_image_int;
char* input_image_char;
bool ascii = false;
int DEFAULT_TILE_WIDTH = 16;
bool DEBUG = true;
int stencil_size = 2;

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
    std::cout << "\nmax_color: " << max_color << "\t cols: " << cols << "\t rows: " << rows << std::endl;

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
    return 0;
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

double testGaussian(std::string in_file, std::string out_file, bool output, int tile_width, int iterations, int iterations_used, std::string file, int kw) {
    int max_color;
    // Read image
   readPGM(in_file, rows, cols, max_color);
    const unsigned int elements = (rows + kw) * (cols + kw);
    const unsigned int stencilmatrixsize = elements * sizeof(int);
    int * gs_image = (int*)malloc(stencilmatrixsize);
    int * result_gs_image = (int*)malloc(stencilmatrixsize);

    if (ascii) {
        printf("ASSCCIII \n");

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
                if (i < (kw/2) || i >= rows || j < (kw/2) || j >= cols){
                    gs_image[(i*(rows+kw)) + j] = 0;
                } else {
                    int offset = ((i-stencil)*cols) + (j-stencil);
                    gs_image[(i*(rows+kw))+ j] = input_image_char[offset] - '0';
                }
            }
        }
    }

    // STart after Timer
    double time = 0.0;
    double start = MPI_Wtime();
    for (int run = 0; run < iterations; ++run) {
       // TODO make multiple GPUs

       int offset = kw/2;
       float weight = 1.0f;
       float sigma = 1;
       float mean = (float)kw/2;

       for (int i = 0; i < rows + kw; i++) {
           for (int j = 0; j < cols + kw; j++) {
               float sum = 0;
               //int offset = tile_width + kw + kw/2 + (i*(cols+kw));
               if (i < (kw/2) || i >= rows || j < (kw/2) || j >= cols){
               } else {
                   int offset = ((i-mean)*cols) + (j-mean);
                   //gs_image[(i*(rows+kw))+ j] = input_image_char[offset] - '0';

                   for (int r = 0; r <= kw; ++r) {
                       for (int c = 0; c <= kw; ++c) {
                           sum += gs_image[(i + r) * (cols+kw) + (j + c)] * std::exp(-0.5 * (std::pow((r-mean)/sigma, 2.0) + std::pow((c-mean)/sigma,2.0))) / (2 * M_PI * sigma * sigma);
                       }
                   }
                   result_gs_image[(i*(rows+kw))+ j] = (int)sum/weight;
               }
           }
       }
       for (int i = 0; i < rows + kw; i++) {
           for (int j = 0; j < cols + kw; j++) {
               float sum = 0;
               //int offset = tile_width + kw + kw/2 + (i*(cols+kw));
               if (i < (kw/2) || i >= rows || j < (kw/2) || j >= cols){
               } else {
                   int offset = ((i-mean)*cols) + (j-mean);
                   //gs_image[(i*(rows+kw))+ j] = input_image_char[offset] - '0';

                   for (int r = 0; r <= kw; ++r) {
                       for (int c = 0; c <= kw; ++c) {
                           sum += result_gs_image[(i + r) * (cols+kw) + (j + c)] * std::exp(-0.5 * (std::pow((r-mean)/sigma, 2.0) + std::pow((c-mean)/sigma,2.0))) / (2 * M_PI * sigma * sigma);
                       }
                   }
                   gs_image[(i*(rows+kw))+ j] = (int)sum/weight;
               }
           }
       }
    }
    double end = MPI_Wtime();

    writePGM(out_file, gs_image, rows+kw, cols+kw, max_color);
    return (end-start);
}

int init(int row, int col)
{
    if (ascii) return input_image_int[row*cols+col];
    else return input_image_char[row*cols+col];
}
int main(int argc, char **argv) {
    std::cout << "\n\n************* Starting the Gaussian Blur *************\n ";

    int nGPUs = 1;
    int nRuns = 1;
    int iterations = MAX_ITER;
    int tile_width = DEFAULT_TILE_WIDTH;
    float cpu_fraction = 0.0;
    //bool warmup = false;
    bool output = false;
    int kw = 2;
    std::string in_file, out_file, file, nextfile; //int kw = 10;
    file = "result_travel.csv";
    if (argc >= 8) {
        nGPUs = atoi(argv[1]);
        nRuns = atoi(argv[2]);
        cpu_fraction = atof(argv[3]);
        if (cpu_fraction > 1) {
            cpu_fraction = 1;
        }
        tile_width = atoi(argv[4]);
        iterations = atoi(argv[5]);

        kw = atoi(argv[7]);

    }

    if (argc == 9) {
        in_file = argv[9];
        size_t pos = in_file.find(".");
        out_file = in_file;
        std::stringstream ss;
        ss << "_" << nGPUs << "_" << iterations <<  "_" << tile_width << "_" << kw << "_gaussian";
        out_file.insert(pos, ss.str());
    } else {
        in_file = "lena.pgm";
        std::stringstream oo;
        oo << in_file << "_" << nGPUs << "_" << iterations <<  "_" << tile_width << "_" << kw << "_gaussian.pgm";
        out_file = oo.str();
    }
    output = true;
    std::stringstream ss;
    ss << file << "_" << iterations;
    nextfile = ss.str();

    int iterations_used = 0;
    double sum = 0.0;
    for (int r = 0; r < nRuns; ++r) {
        sum += testGaussian(in_file, out_file, output, tile_width, iterations, iterations_used, nextfile, kw);
    }

    if (output) {
        std::ofstream outputFile;
        outputFile.open(nextfile, std::ios_base::app);
        outputFile << "" + std::to_string(nGPUs) + ";" + std::to_string(tile_width) +";" + std::to_string(iterations) + ";" +
        std::to_string(iterations_used) + ";" + std::to_string(sum/nRuns) + ";\n";
        outputFile.close();
    }
    std::cout << "\n************* Finished the Gaussian Blur *************\n ";

    return 0;
}
