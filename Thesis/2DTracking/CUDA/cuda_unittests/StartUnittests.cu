#include <gtest/gtest.h>

int main(int argc, char** argv)
{
    // Taken from Nvidias CUDA examples.
    int devID;
    cudaDeviceProp props;
    
    // get number of SMs on this GPU
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&props, devID);
    
    printf("Device %d: \"%s\" with Compute %d.%d capability\n", 
           devID, props.name, props.major, props.minor);
    printf("Processor Count on GPU: %i\n\n", props.multiProcessorCount);

    testing::InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();
    
    return 0;
}
    
    
    
