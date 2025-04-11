#include <chrono>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <random>
#include <time.h>

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#endif

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_ENABLE_EXCEPTIONS

#include "CL/opencl.hpp"

#ifndef ANALYZE
#define ANALYZE 1
#endif

#define dbgs                                                                   \
  if (!ANALYZE) {                                                              \
  } else                                                                       \
    std::cout

constexpr size_t ARR_SIZE = 1025;
constexpr size_t LOCAL_SIZE = 1;

#define STRINGIFY(...) #__VA_ARGS__

// ---------------------------------- OpenCL ---------------------------------
const char *vakernel = STRINGIFY(__kernel void bitonic_sort(
    __global int *A, const uint curr_stage, const uint curr_pass, const uint direction) {
  
  uint i = get_global_id(0);
	
  uint dir = direction;
	uint shift = curr_stage - curr_pass;
	uint distance = 1 << shift;
	uint left_mask = distance - 1;
	uint direction_mask = 1 << curr_stage;

	uint left_index = ((i >> shift) << (shift + 1)) + (i & left_mask);
	uint right_index = left_index + distance;

	uint left = A[left_index];
	uint right = A[right_index];
	uint bigger, smaller;

	if (left > right) {
		bigger = left;
		smaller = right;
	}
	else {
		bigger = right;
		smaller = left;
	}

	if ((i & direction_mask) == direction_mask) dir = !dir;

	if (dir) {
		A[left_index] = smaller;
		A[right_index] = bigger;
	}
	else {
		A[left_index] = bigger;
		A[right_index] = smaller;
	}
});
// ---------------------------------- OpenCL ---------------------------------

// OpenCL application encapsulates platform, context and queue
// We can offload vector addition through its public interface
class OclApp {
  cl::Platform P_;
  cl::Context C_;
  cl::CommandQueue Q_;

  static cl::Platform select_platform();
  static cl::Context get_gpu_context(cl_platform_id);

  using vadd_t = cl::KernelFunctor<cl::Buffer, const uint, const uint, const uint>;

public:
  OclApp() : P_(select_platform()), C_(get_gpu_context(P_())), Q_(C_) {
    cl::string name = P_.getInfo<CL_PLATFORM_NAME>();
    cl::string profile = P_.getInfo<CL_PLATFORM_PROFILE>();
    dbgs << "Selected: " << name << ": " << profile << std::endl;
  }

  // C[i] = A[i] + B[i]
  // Here we shall ask ourselfes: why not template?
  void vadd(cl_int const *A, cl_int *C, size_t Sz);
};

// select first platform with some GPUs
cl::Platform OclApp::select_platform() {
  cl::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  for (auto p : platforms) {
    // note: usage of p() for plain id
    cl_uint numdevices = 0;
    ::clGetDeviceIDs(p(), CL_DEVICE_TYPE_GPU, 0, NULL, &numdevices);
    if (numdevices > 0)
      return cl::Platform(p); // retain?
  }
  throw std::runtime_error("No platform selected");
}

// get context for selected platform
cl::Context OclApp::get_gpu_context(cl_platform_id PId) {
  cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(PId),
      0 // signals end of property list
  };

  return cl::Context(CL_DEVICE_TYPE_GPU, properties);
}

void OclApp::vadd(cl_int const *APtr, cl_int *CPtr, size_t Sz) {
  size_t BufSz = Sz * sizeof(cl_int);

  cl::Buffer A(C_, CL_MEM_READ_WRITE, BufSz);

  cl::copy(Q_, APtr, APtr + Sz, A);

  // printf("building\n");
  cl::Program program(C_, vakernel, true /* build immediately */);
  // printf("builded\n");

  vadd_t add_vecs(program, "bitonic_sort");

  cl::NDRange GlobalRange(Sz);
  cl::NDRange LocalRange(LOCAL_SIZE);
  cl::EnqueueArgs Args(Q_, GlobalRange, LocalRange);

  cl_uint total_stages = 0;
  uint direction = 1;


  for(size_t i = Sz; i > 1; i >>= 1){
    ++total_stages;
  }

  for(cl_uint curr_stage = 0; curr_stage < total_stages; ++curr_stage){
    for(cl_uint curr_pass = 0; curr_pass < curr_stage + 1; ++curr_pass){
      cl::Event evt = add_vecs(Args, A, curr_stage, curr_pass, direction);
      evt.wait();
    }
  }

  printf("\n");
  cl::copy(Q_, A, CPtr, CPtr + Sz);
}

int main() try {
  OclApp app;

  size_t count = 0;
  std::cin >> count;
  uint32_t magnitude = 0;
  uint arr_size = 1;
  while(arr_size < count){
    magnitude++;
    arr_size = 1 << magnitude;
  }

  printf("%d %d\n", arr_size, magnitude);

  cl::vector<cl_int> src(arr_size), dst(arr_size);

  // std::random_device dev;
  // std::mt19937 rng(dev());
  // std::uniform_int_distribution<std::mt19937::result_type> dist100(1,1000);

  
  for(size_t i = 0; i < count; ++i){
    // src[i] = dist100(rng);
    std::cin >> src[i];
  }
  for(size_t i = count; i < arr_size; ++i){
    src[i] = 0x7FFFFFFF;
  }

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  app.vadd(src.data(), dst.data(), dst.size());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  
  std::cout << "Time difference in bitonic = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;

  for(size_t i = 0; i < arr_size; ++i){
    if(dst[i] > dst[i+1]){
      std::cout << "Bad sort!\n";
    }
  }

  for(size_t i = 0; i < arr_size-1; ++i){
    std::cout << dst[i] << ' ';
  }
  std::cout << '\n';

  std::cout << "OK\n";
  return 0;

  begin = std::chrono::steady_clock::now();
  std::qsort
  (
      src.data(),
      src.size(),
      sizeof(decltype(src)::value_type),
      [](const void* x, const void* y)
      {
          const int arg1 = *static_cast<const int*>(x);
          const int arg2 = *static_cast<const int*>(y);
          const auto cmp = arg1 <= arg2;
          if (cmp < 0)
              return -1;
          if (cmp > 0)
              return 1;
          return 0;
      }
  );
  end = std::chrono::steady_clock::now();
  std::cout << "Time difference in qsort = " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << "[ms]" << std::endl;



} catch (cl::Error &err) {
  std::cerr << "OCL ERROR " << err.err() << ":" << err.what() << std::endl;
  return -1;
} catch (std::runtime_error &err) {
  std::cerr << "RUNTIME ERROR " << err.what() << std::endl;
  return -1;
} catch (...) {
  std::cerr << "UNKNOWN ERROR\n";
  return -1;
}