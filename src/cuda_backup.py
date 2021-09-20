@cuda.jit
def convolution_kernel(input, weights, output, kernel, layer_depth, strides, dilation, z_padding, padding):

    x = cuda.threadIdx.x
    d = cuda.blockIdx.x
    k = cuda.blockIdx.y
    tpb = cuda.blockDim.x

    input_length = input.shape[0]
    
    s_input = input[:,d]
    s_weights = weights[:,d,k]

    #Absolute postion of thread in grid
    x = cuda.threadIdx.x
    d = cuda.blockIdx.x
    k = cuda.blockIdx.y
    tpb = cuda.blockDim.x

    thread_offset = -(-output.shape[0]//tpb)

    i = x*thread_offset
    while i < (x+1)* thread_offset:

        if i >= output.shape[0]:
            # Quit if x is outside of of valid ouput boundary
            return
        offset = i*strides-padding
        j = 0
        while j < kernel:
            if((offset+j*dilation)/(z_padding+1) < input_length and (offset+j*dilation)%(z_padding+1) == 0): #in range(input.shape[0])
                tmp = s_weights[j] * s_input[int((offset+j*dilation)/(z_padding+1))]
                output[i,k,d] += tmp               
            j +=1
        i += 1


def convolution_cuda(input, weights, bias,  kernel, layer_depth, strides, dilation, z_padding, padding, a, threads_per_block):
    input_length, input_depth = input.shape[0], input.shape[1]
    output_length = floor((input_length+2*padding+(input_length-1)*z_padding+a-(kernel+(kernel-1)*(dilation-1)))/strides)+1
    output = np.zeros((output_length, layer_depth, input_depth))

    input_global_mem = cuda.to_device(input)
    weights_global_mem = cuda.to_device(weights)
    output_global_mem = cuda.to_device(output)

    bpg =  (input_depth, layer_depth)
    convolution_kernel[bpg,threads_per_block](input_global_mem, weights_global_mem, output_global_mem, kernel, layer_depth, strides, dilation, z_padding, padding)
    output_d = output_global_mem.copy_to_host()
    output = np.sum(output_d, axis=2) + bias
    return output