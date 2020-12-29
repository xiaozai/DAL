#include <THC/THC.h>
#include <math.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "depthconv_cuda_kernel.cuh"
extern THCState *state;

int depthconv_forward_cuda(at::Tensor &input, at::Tensor &input_depth, at::Tensor &weight, at::Tensor &bias, at::Tensor &output,
                             at::Tensor &columns, at::Tensor &ones, int kW,
                             int kH, int dW, int dH, int padW, int padH,
                             int dilationH, int dilationW) {
  cudaStream_t stream=at::cuda::getCurrentCUDAStream();
  int batch = 1;

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);
  long nOutputPlane = weight.size(0);
  long outputWidth =(inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =(inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  for (int elt = 0; elt < batchSize; elt++) {
    at::Tensor input_n = at::select(input,0,elt);
    at::Tensor depth_n = at::select(input_depth,0,elt);
    at::Tensor output_n =at::select(output,0,elt);
    // Do bias first
    long m_ = nOutputPlane;
    long n_ = outputHeight * outputWidth;
    long k_ = 1;
    at::zero_(output_n);
    depthconv_im2col(
        stream, input_n.data<float>(), depth_n.data<float>(), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, columns.data<float>());

    long m = nOutputPlane;
    long n = columns.size(1);
    long k = nInputPlane * kH * kW;

    THCudaBlas_Sgemm(state, 'n', 'n', n, m, k, 1.0f,
                     columns.data<float>(), n,
                     weight.data<float>(), k, 1.0f,
                     output_n.data<float>(), n);
  }
  return 1;
}

int depthconv_backward_input_cuda(
    at::Tensor &input, at::Tensor &input_depth, at::Tensor &gradOutput,
    at::Tensor &gradInput, at::Tensor &weight,
    at::Tensor &columns, int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationH, int dilationW) {

  cudaStream_t stream=at::cuda::getCurrentCUDAStream();
  int batch = 1;
  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);
  long nOutputPlane = weight.size(0);
  long outputWidth =(inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =(inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
//  printf("columns size: %d,%d\n", columns->size[0],columns->size[1]);


  for (int elt = 0; elt < batchSize; elt++) {
    at::Tensor gradInput_n = at::select(gradInput,0,elt);
    at::Tensor input_depth_n = at::select(input_depth,0,elt);
    at::Tensor gradOutput_n =at::select(gradOutput,0,elt);

    long m = nInputPlane * kW * kH;
    long n = columns.size(1);
    long k = nOutputPlane;

    THCudaBlas_Sgemm(state, 'n', 't', n, m, k, 1.0f,
                     gradOutput_n.data<float>(), n,
                     weight.data<float>(), m, 0.0f,
                     columns.data<float>(), n);

    depthconv_col2im(
        stream, columns.data<float>(),
        input_depth_n.data<float>(), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, gradInput_n.data<float>());
  }
  return 1;
}


int depthconv_backward_parameters_cuda(
    at::Tensor &input, at::Tensor &input_depth, at::Tensor &gradOutput,
    at::Tensor &gradWeight, at::Tensor &gradBias,
    at::Tensor &columns, at::Tensor &ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationH, int dilationW,
    float scale) {

  cudaStream_t stream=at::cuda::getCurrentCUDAStream();
  int batch = 1;
  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);
  long nOutputPlane = gradWeight.size(0);
  long outputWidth =(inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =(inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;


  for (int elt = 0; elt < batchSize; elt++) {
    at::Tensor input_n = at::select(input,0,elt);
    at::Tensor depth_n = at::select(input_depth,0,elt);
    at::Tensor gradOutput_n =at::select(gradOutput,0,elt);

    depthconv_im2col(
        stream, input_n.data<float>(),
        depth_n.data<float>(), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, columns.data<float>());

    long m = nOutputPlane;
    long n = nInputPlane * kW * kH;
    long k = columns.size(1);

    THCudaBlas_Sgemm(state, 't', 'n', n, m, k, scale,
                     columns.data<float>(), k,
                     gradOutput_n.data<float>(), k, 1.0f,
                     gradWeight.data<float>(), n);
    // Do Bias:
    // M,N,K are dims of matrix A and B
    // long m_ = nOutputPlane;
    // long k_ = outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    // if (gradBias)
    //     THCudaBlas_Sgemv(
    //       state,
    //       't',
    //       k_, m_,
    //       scale,
    //       THCudaTensor_data(state, gradOutput_n), k_,
    //       THCudaTensor_data(state, ones), 1, 1.0f,
    //       THCudaTensor_data(state, gradBias), 1);
  }
  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depthconv_forward_cuda", &depthconv_forward_cuda, "DepthConv_forward");
    m.def("depthconv_backward_input_cuda", &depthconv_backward_input_cuda, "DepthConv_backward_input");
    m.def("depthconv_backward_parameters_cuda", &depthconv_backward_parameters_cuda, "DepthConv_backward_parameters");
}
