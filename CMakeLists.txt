cmake_minimum_required (VERSION 3.10.2)
project(Muesli)
enable_language(CUDA CXX)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_HOST_FLAGS " -Xcompiler ")
set(CMAKE_CUDA_HOST_LINKER_FLAGS " -Xlinker ")

set(CMAKE_CXX_FLAGS_BENCHMARKPALMA "-O3 -DNDEBUG -march=broadwell -DMPICH_IGNORE_CXX_SEEK -std=c++14")

string(REPLACE " " "," MPI_CXX_LINK_FLAG ${MPI_CXX_LINK_FLAGS})
    
foreach (flag ${MPI_CXX_COMPILE_OPTIONS})
  string(APPEND CMAKE_CUDA_HOST_FLAGS ",${flag}")
endforeach (flag ${MPI_CXX_COMPILE_OPTIONS})
foreach (flag ${MPI_CXX_LINK_FLAG})
  string(APPEND CMAKE_CUDA_HOST_LINKER_FLAGS ",${flag}")
endforeach (flag ${MPI_CXX_LINK_FLAG})

foreach (flag ${OpenMP_CXX_FLAGS})
  string(APPEND CMAKE_CUDA_HOST_FLAGS ",${flag}")
  string(APPEND CMAKE_CUDA_HOST_LINKER_FLAGS ",${flag}")
endforeach (flag ${OpenMP_CXX_FLAGS})

string(APPEND CMAKE_CUDA_FLAGS ${CMAKE_CUDA_HOST_FLAGS})
string(APPEND CMAKE_CUDA_FLAGS ${CMAKE_CUDA_HOST_LINKER_FLAGS})

set( CMAKE_CUDA_FLAGS_BENCHMARKPALMA "-O3 -arch=compute_35 --expt-extended-lambda -code=sm_35 -use_fast_math -w -restrict -Xptxas -O3 -Xcompiler -O3,-DNDEBUG,-march=broadwell,-DMPICH_IGNORE_CXX_SEEK,-std=c++14")

# add muesli libs

#set(MUESLI_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/muesli-${PROJECT_VERSION})
#set(MUESLI_INCLUDE_INSTALL_DIR ${MUESLI_INSTALL_DIR}/include)
#set(MUESLI_INCLUDE_DETAIL_INSTALL_DIR ${MUESLI_INCLUDE_INSTALL_DIR}/detail)
## add_library(muesli STATIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/detail ${PROJECT_SOURCE_DIR}/src)

##  $<BUILD_INTERFACE:${MPI_CXX_INCLUDE_PATH}>
##  $<INSTALL_INTERFACE:${MUESLI_INCLUDE_INSTALL_DIR}>
##  $<INSTALL_INTERFACE:${MUESLI_INCLUDE_INSTALL_DIR}/detail>

## install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src DESTINATION ${MUESLI_INSTALL_DIR})

set(Example main.cu)
## set(Example examples/mandelbrotDA.cu)
## set(Example examples/piDA.cu)
##set(Example examples/dm_test.cu)

### get name of current example
get_filename_component(example_name ${Example} NAME_WE)
add_executable(${example_name} ${Example})

target_compile_features(${example_name} PUBLIC cxx_std_14)
target_include_directories(${example_name} 
  PUBLIC ${MPI_CXX_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/src
  PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/detail)
target_compile_definitions(${example_name} PUBLIC ${MPI_CXX_COMPILE_DEFINITIONS})
target_compile_options(${example_name} PUBLIC )
target_link_libraries(${example_name} PUBLIC ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})

