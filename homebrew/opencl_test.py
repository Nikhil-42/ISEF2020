import pyopencl as cl
from pyopencl import cltypes
import numpy

def add_connection(to, frm, weights):
    weights[to, frm] = numpy.random.random(1)[0]
 
if __name__ == "__main__":
    node_count = 5

    numpy.random.seed(12)

    nodes = numpy.zeros((node_count, 3)).astype(numpy.float32)
    frm = numpy.array([[-1, -1, -1, -1, 0],[-1, -1, -1, -1, 0],[0, 1, -1, -1, 2],[0, 1, -1, -1, 2],[2, 3, -1, -1, 2]], dtype=numpy.int32)
    weights = numpy.random.random((node_count, node_count)).astype(numpy.float32)

    layer = [[2, 3],[4]]

    add_connection(2, 0, weights)
    add_connection(2, 1, weights)
    add_connection(3, 0, weights)
    add_connection(3, 1, weights)
    add_connection(4, 2, weights)
    add_connection(4, 3, weights)

    ## Step #1. Obtain an OpenCL platform.
    platform = cl.get_platforms()[0]
     
    ## It would be necessary to add some code to check the check the support for
    ## the necessary platform extensions with platform.extensions
     
    ## Step #2. Obtain a device id for at least one device (accelerator).
    device = platform.get_devices()[0]
     
    ## It would be necessary to add some code to check the check the support for
    ## the necessary device extensions with device.extensions
     
    ## Step #3. Create a context for the selected device.
    context = cl.Context([device])
     
    ## Step #4. Create the accelerator program from source code.
    ## Step #5. Build the program.
    ## Step #6. Create one or more kernels from the program functions.
    network_operations = cl.Program(context, """
        __kernel void forward_propagate(const int node_count, __global float* nodes, __global const int* from,
        __global const float* weights, __global const int* layer, __global float* raw_activations)
        {
            int gid = get_global_id(0);
            __local int current_node;
            current_node = layer[gid];
            __local float sum;
            sum = 0;
            __local int from_node; 
            for(int i = 0; i < from[(current_node+1)*node_count-1]; i++) {
                from_node = from[current_node*node_count+i];
                nodes[current_node*3 + i+1] = weights[current_node*node_count+from_node];
                sum += nodes[from_node*3] * weights[current_node*node_count+ from_node];
            }
            
            raw_activations[gid] = sum;
        }
        """).build()

    activation_functions = cl.Program(context, """
        __kernel void relu(__global const float* raw_activations, __global const int* layer, __global float* nodes) {
            int gid = get_global_id(0);
            float activation = raw_activations[gid];
            if (activation > 0) {
                nodes[layer[gid]*3] = activation;
            } else {
                nodes[layer[gid]] = 0;
            }
        }
    """).build()
     
    ## Step #7. Create a command queue for the target device.
    queue = cl.CommandQueue(context)
     
    ## Step #8. Allocate device memory and move input data from the host to the device memory.
    mem_flags = cl.mem_flags

    nodes[0, 0] = 0
    nodes[1, 0] = 1
    weights[2, 0] = -0.81781853
    weights[2, 1] = 0.48803631
    weights[3, 0] = 0.71323677
    weights[3, 1] = -0.71286155
    weights[4, 2] = 2.04849235
    weights[4, 3] = 1.40170791

    print(weights)


    nodes_buf = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.USE_HOST_PTR, hostbuf=nodes)
    from_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=frm)
    weights_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=weights)
    ## Step #9. Associate the arguments to the kernel with kernel object.
    ## Step #10. Deploy the kernel for device execution.
    this_layer = numpy.array(layer[0], dtype=numpy.int32)
    layer_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=this_layer)
    raw_activations_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, 4*len(this_layer))
    network_operations.forward_propagate(queue, this_layer.shape, None, numpy.int32(node_count), nodes_buf, from_buf, weights_buf, layer_buf, raw_activations_buf)
    activation_functions.relu(queue, this_layer.shape, None, raw_activations_buf, layer_buf, nodes_buf)
    
    this_layer = numpy.array(layer[1], dtype=numpy.int32)
    layer_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=this_layer)
    raw_activations_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, 4*len(this_layer))
    network_operations.forward_propagate(queue, this_layer.shape, None, numpy.int32(node_count), nodes_buf, from_buf, weights_buf, layer_buf, raw_activations_buf)
    activation_functions.relu(queue, this_layer.shape, None, raw_activations_buf, layer_buf, nodes_buf)
    ## Step #11. Move the kernel’s output data to host memory.
    cl.enqueue_copy(queue, nodes, nodes_buf)
    ## Step #12. Release context, program, kernels and memory.
    ## PyOpenCL performs this step for you, and therefore,
    ## you don't need to worry about cleanup code

    print(nodes)