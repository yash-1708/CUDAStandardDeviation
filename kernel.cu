
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <ctime>
#include <math.h>

#define M 2048//threads per block

__global__ void blockSummer(int *mainVec,int *blockSum,int N,int mean,bool mode){
	__shared__ int sumVec[M+1];
	int tid = threadIdx.x;
	int eid = blockIdx.x * blockDim.x + threadIdx.x;
	if (eid < N) {
		if (mode == true) {
			sumVec[tid] = (mainVec[eid] - mean) * (mainVec[eid] - mean);
		}
		else {
			sumVec[tid] = mainVec[eid];
		}
	}
	else {
		sumVec[tid] = 0;
	}
	__syncthreads();
	for (unsigned int i = 1; i < blockDim.x; i *= 2) {
		if (tid % (2 * i) == 0) {
			sumVec[tid] += sumVec[tid + i];
		}
		__syncthreads();
	}
	if (tid == 0) {
		blockSum[blockIdx.x] = sumVec[0];
	}
}

void serialStdDev(int* mainVec, int* intermediate, int N) {
	int sum = 0;
	for (size_t i = 0; i < N; i++) {
		sum += mainVec[i];
	}

	int mean = sum / N;

	for (size_t i = 0; i < N; i++) {
		intermediate[i] = (mainVec[i] - mean) * (mainVec[i] - mean);
	}

	int sum2 = 0;
	
	for (size_t i = 0; i < N; i++) {
		sum2 += intermediate[i];
	}

	float mean2 = sum2 / N;
	float stddev = sqrt(mean2);
	//printf("\nStandard Deviation : %f",stddev);
}

void randomInts(int* vector, int length) {
	for (size_t i = 0; i < length; i++){
		vector[i] = rand() % 100;
	}
}

int main(){
	clock_t start, stop;
	int N = 95000000;
	int numOfBlocks = (N + M - 1) / M;
	int *mainVec = (int*)malloc(N * sizeof(int));
	int* intermediate = (int*)malloc(N * sizeof(int));
	int *blockSum = (int*)malloc(numOfBlocks * sizeof(int));
	int *blockSum2 = (int*)malloc(numOfBlocks * sizeof(int));
	randomInts(mainVec, N);
	int *d_mainVec, * d_blockSum, * d_blockSum2;
	cudaMalloc(&d_mainVec, N * sizeof(int));
	cudaMalloc(&d_blockSum, numOfBlocks * sizeof(int));

	//printf("\n*************PARALLEL EXECUTION*************/n");

	cudaMemcpy(d_mainVec, mainVec, N * sizeof(int), cudaMemcpyHostToDevice);
	start = std::clock();
	blockSummer <<<numOfBlocks, M >>> (d_mainVec, d_blockSum, N, 0, false);
	cudaDeviceSynchronize();
	cudaMemcpy(blockSum, d_blockSum, numOfBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	int sum = 0;
	for (size_t i = 0; i < numOfBlocks; i++){
		sum += blockSum[i];
	}
	int mean = (int)(sum / N);
	blockSummer <<<numOfBlocks, M >>> (d_mainVec, d_blockSum, N, mean, true);
	cudaDeviceSynchronize();
	cudaMemcpy(blockSum, d_blockSum, numOfBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	int stddevsum = 0;
	for (size_t i = 0; i < numOfBlocks; i++) {
		stddevsum += blockSum[i];
	}
	float stddev = (stddevsum / N);
	float finalstddev = sqrt(stddev);
	//printf("\nStandard Deviation : %f", finalstddev);
	stop = std::clock();
	long float timeP = stop - start;
	//printf("\n*************SERIAL EXECUTION*************/n");
	start = std::clock();
	serialStdDev(mainVec, intermediate, N);
	stop = std::clock();
	long float timeN = stop - start;

	//getting GPU properties and storing in prop
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int cores = prop.multiProcessorCount * 128;
	float totalCost = cores * timeP;

	//results printing
	printf("\n********************************************************************************************************\n");
	printf("N \t\t\t Nor Time \t Par Time \t Cores \t\t Tot Cost \t Speedup \t Efficiency \n");
	printf("%-20d \t %-7.3f \t %-7.3f \t %-10d \t %-7.3f \t %-7.3f \t %-5.5f \n", N, timeN, timeP, cores, totalCost, timeN / timeP, timeN / (timeP * cores));
	printf("\n********************************************************************************************************\n");

	free(mainVec);
	free(blockSum);
	free(intermediate);
	cudaFree(d_mainVec);
	cudaFree(d_blockSum);

	return 0;
}
