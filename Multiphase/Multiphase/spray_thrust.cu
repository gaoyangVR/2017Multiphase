#include <Windows.h>
#include<cuda.h>
#include <GL/freeglut.h>
#include<cuda_gl_interop.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include <thrust/scan.h>
typedef unsigned int uint;

void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
{
	thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
		thrust::device_ptr<uint>(dGridParticleHash + numParticles),
		thrust::device_ptr<uint>(dGridParticleIndex));
}

//MC
void ThrustScanWrapper(unsigned int* output, unsigned int* input, unsigned int numElements)
{
	thrust::exclusive_scan(thrust::device_ptr<unsigned int>(input),
		thrust::device_ptr<unsigned int>(input + numElements),
		thrust::device_ptr<unsigned int>(output));
}