inline signed char abs_device(signed char src) {
  if (src == -128) {
    return 127;
  }
  else {
    return abs((int)src);
  }
}

inline float abs_device(float src) {
  if (src >= 0.f) {
    return src;
  }
  else {
    return (0.f - src);
  }
  // return fabs(src);
}

__kernel
void absU8Kernel3(global const signed char* src, int rows, int cols,
                  int src_stride, global signed char* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const char* input = (char*)(src + element_y * src_stride);
  char4 input_value = vload4(element_x, input);

  char4 output_value;
  output_value.x = abs_device(input_value.x);
  output_value.y = abs_device(input_value.y);
  output_value.z = abs_device(input_value.z);
  output_value.w = abs_device(input_value.w);

  char* output = (char*)(dst + element_y * dst_stride);
  vstore4(output_value, element_x, output);
}

__kernel
void absU8Kernel0(global const signed char* src, int rows, int cols,
                  int src_stride, global signed char* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const char4* input = (char4*)(src + element_y * src_stride);
  char4 input_value = input[element_x];

  char4 output_value;
  output_value.x = abs_device(input_value.x);
  output_value.y = abs_device(input_value.y);
  output_value.z = abs_device(input_value.z);
  output_value.w = abs_device(input_value.w);

  char4* output = (char4*)(dst + element_y * dst_stride);
  output[element_x] = output_value;
}

__kernel
void absU8Kernel1(global const signed char* src, int cols,
                  global signed char* dst) {
  int element_x = get_global_id(0);
  int index_x = element_x << 2;
  if (index_x >= cols) {
    return;
  }

  const char4* input = (char4*)src;
  char4 input_value, output_value;
  input_value = input[element_x];

  if (index_x < cols - 3) {
    output_value.x = abs_device(input_value.x);
    output_value.y = abs_device(input_value.y);
    output_value.z = abs_device(input_value.z);
    output_value.w = abs_device(input_value.w);

    char4* output = (char4*)dst;
    output[element_x] = output_value;
  }
  else {
    output_value.x = abs_device(input_value.x);
    if (index_x < cols - 1) {
      output_value.y = abs_device(input_value.y);
    }
    if ((index_x < cols - 2)) {
      output_value.z = abs_device(input_value.z);
    }

    dst[index_x] = output_value.x;
    if (index_x < cols - 1) {
      dst[index_x + 1] = output_value.y;
    }
    if ((index_x < cols - 2)) {
      dst[index_x + 2] = output_value.z;
    }
  }
}

__kernel
void absU8Kernel2(global const signed char* src, int rows, int cols,
                  int src_stride, global signed char* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 2;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  const signed char* input = src + element_y * src_stride;
  signed char* output = dst + element_y * dst_stride;

  signed char input_value0, input_value1, input_value2, input_value3;
  signed char output_value0, output_value1, output_value2, output_value3;

  if (index_x < cols - 3) {
    input_value0 = input[index_x];
    input_value1 = input[index_x + 1];
    input_value2 = input[index_x + 2];
    input_value3 = input[index_x + 3];

    output_value0 = abs_device(input_value0);
    output_value1 = abs_device(input_value1);
    output_value2 = abs_device(input_value2);
    output_value3 = abs_device(input_value3);

    output[index_x]     = output_value0;
    output[index_x + 1] = output_value1;
    output[index_x + 2] = output_value2;
    output[index_x + 3] = output_value3;
  }
  else {
    input_value0 = input[index_x];
    if (index_x < cols - 1) {
      input_value1 = input[index_x + 1];
    }
    if ((index_x < cols - 2)) {
      input_value2 = input[index_x + 2];
    }

    output_value0 = abs_device(input_value0);
    if (index_x < cols - 1) {
      output_value1 = abs_device(input_value1);
    }
    if ((index_x < cols - 2)) {
      output_value2 = abs_device(input_value2);
    }

    output[index_x] = output_value0;
    if (index_x < cols - 1) {
      output[index_x + 1] = output_value1;
    }
    if ((index_x < cols - 2)) {
      output[index_x + 2] = output_value2;
    }
  }
}

__kernel
void absF32Kernel0(global const float* src, int rows, int cols, int src_stride,
                   global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const float2* input = (float2*)((uchar*)src + element_y * src_stride);
  float2 input_value = input[element_x];

  float2 output_value;
  output_value.x = abs_device(input_value.x);
  output_value.y = abs_device(input_value.y);

  float2* output = (float2*)((uchar*)dst + element_y * dst_stride);
  output[element_x] = output_value;
}
__kernel
void absF32Kernel1(global const float* src, int rows, int cols, int src_stride,
                   global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  int index_x = element_x << 1;
  if (element_y >= rows || index_x >= cols) {
    return;
  }

  const float2* input = (float2*)((uchar*)src + element_y * src_stride);
  float2 input_value = input[element_x];
  float2 output_value;

  if (index_x < cols - 1) {
    output_value.x = abs_device(input_value.x);
    output_value.y = abs_device(input_value.y);

    float2* output = (float2*)((uchar*)dst + element_y * dst_stride);
    output[element_x] = output_value;
  }
  else {
    output_value.x = abs_device(input_value.x);
    if (index_x < cols - 1) {
      output_value.y = abs_device(input_value.y);
    }

    float* output = (float*)((uchar*)dst + element_y * dst_stride);
    output[index_x]     = output_value.x;
    output[index_x + 1] = output_value.y;
  }
}

__kernel
void absF32Kernel2(global const float* src, int rows, int cols, int src_stride,
                   global float* dst, int dst_stride) {
  int element_x = get_global_id(0);
  int element_y = get_global_id(1);
  if (element_y >= rows || element_x >= cols) {
    return;
  }

  const float* input = (float*)(src + element_y * src_stride);
  float4 input_value = vload4(element_x, input);

  float4 output_value;
  output_value.x = abs_device(input_value.x);
  output_value.y = abs_device(input_value.y);
  output_value.z = abs_device(input_value.z);
  output_value.w = abs_device(input_value.w);

  float* output = (float*)(dst + element_y * dst_stride);
  vstore4(output_value, element_x, output);
}
