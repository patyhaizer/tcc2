#include "configuration.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <assert.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

using namespace std;

struct Instance {
    //char m_alphabet[ALPHABET_SIZE + 1];
    char m_instanceData[INSTANCE_SIZE][CHROMOSSOME_SIZE + 1];
};

struct Individual {
    double m_chromossome[CHROMOSSOME_SIZE];
    int m_fitness;
};

__host__ __device__
bool operator < (const Individual& lhs, const Individual& rhs)
{
   return lhs.m_fitness < rhs.m_fitness;
}

struct Population {
    Individual m_chromossomes[BRKGA_pop_size];
};

///////////////////////////////// CUDA FUNCTIONS - BEGIN //////////////////////////////////////////////////////
__device__
char decodeAlele(const double& randomValue) {
    // char returnValue;
    // if (randomValue <= 0.50) {
    //     int t = (int) floor(((randomValue * 100) - 5) / 5);
    //     returnValue = t;
    // }
    // return returnValue;
    if(randomValue <= 0.05){
        return '0';
    } else if(randomValue <= 0.10) {
        return '1';
    } else if(randomValue <= 0.15) {
        return '2';
    } else if(randomValue <= 0.20) {
        return '3';
    } else if(randomValue <= 0.25) {
        return '4';
    } else if(randomValue <= 0.30) {
        return '5';
    } else if(randomValue <= 0.35) {
        return '6';
    } else if(randomValue <= 0.40) {
        return '7';
    } else if(randomValue <= 0.45) {
        return '8';
    } else if(randomValue <= 0.50) {
        return '9';
    } else if(randomValue <= 0.55) {
        return 'a';
    } else if(randomValue <= 0.60) {
        return 'b';
    } else if(randomValue <= 0.65) {
        return 'c';
    } else if(randomValue <= 0.70) {
        return 'd';
    } else if(randomValue <= 0.75) {
        return 'e';
    } else if(randomValue <= 0.80) {
        return 'f';
    } else if(randomValue <= 0.85) {
        return 'g';
    } else if(randomValue <= 0.90) {
        return 'h';
    } else if(randomValue <= 0.95) {
        return 'i';
    } else {
        return 'j';
    }

    // if(randomValue <= 0.25){
    //     return 'A';
    // } else if(randomValue <= 0.50) {
    //     return 'C';
    // } else if(randomValue <= 0.75) {
    //     return 'G';
    // } else {
    //     return 'T';
    // }
}

__global__
void setupRand(curandState * state, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &state[tid]);
}

__global__ void genInitialPop(curandState * globalState, Population *pop) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int individualId = blockIdx.x;
    int aleloId = threadIdx.x;
    curandState localState = globalState[tid];
    double random = curand_uniform_double(&localState);
    if (aleloId < CHROMOSSOME_SIZE) {
        pop->m_chromossomes[individualId].m_chromossome[aleloId] = random;
        pop->m_chromossomes[individualId].m_fitness = 0;
    }
    globalState[tid] = localState;
}

__global__ void evaluate(Population *pop, Instance * inst) {
    int individualId = blockIdx.x;
    int aleloId = threadIdx.x;
    //int instanceId = blockIdx.y;
    char decodedAlele;
    __shared__ int chromossomeCost[INSTANCE_SIZE][CHROMOSSOME_SIZE];
    int individualFitness;

    if (aleloId < CHROMOSSOME_SIZE) {
        // DO I NEED INITIALIZATION HERE?????
        //chromossomeCost[aleloId] = 0;
        //__syncthreads();
        decodedAlele = decodeAlele(pop->m_chromossomes[individualId].m_chromossome[aleloId]);
        for (int i = 0; i < INSTANCE_SIZE; ++i) {
            chromossomeCost[i][aleloId] = (decodedAlele != inst->m_instanceData[i][aleloId]);
        }
        __syncthreads();
        if (threadIdx.x < 9) {
            individualFitness = 0;
            // Sum all elements to calculate the cost
            for (int i = 0; i < CHROMOSSOME_SIZE; ++i) {
                individualFitness += chromossomeCost[threadIdx.x][i];
            }
            // Access global memory to do an atomic operation to uptade chromossome cost
            atomicMax(&(pop->m_chromossomes[individualId].m_fitness), individualFitness);
        }
    }
}

__global__ void generateNextPop(curandState * globalState, Population *prevPop, Population *nextPop) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state = globalState[tid];

    __shared__ int eliteId;
    __shared__ int nonEliteId;

    int individualId = blockIdx.x;
    int aleloId = threadIdx.x;

    if (blockIdx.x < ELITE_SIZE) {
        // then make the copy of the chromossome since it is an elite
        nextPop->m_chromossomes[individualId].m_chromossome[aleloId] = prevPop->m_chromossomes[individualId].m_chromossome[aleloId];
        nextPop->m_chromossomes[individualId].m_fitness = 0;
    } else if (blockIdx.x < ELITE_SIZE + MUTANT_SIZE) {
        // generate mutant individuals
        double random = curand_uniform_double(&state);
        nextPop->m_chromossomes[individualId].m_chromossome[aleloId] = random;
        nextPop->m_chromossomes[individualId].m_fitness = 0;
    } else {
        // Only thread 0 calculates the individuals which will crossover
        // The individuals are the same for the whole block - one block is one chromossome
        if (threadIdx.x == 0) {
            // create offspring
            // 1 - choose one elite individual
            eliteId = (int) (curand_uniform(&state) * (ELITE_SIZE - 1));
            // 2 - choose non-elite individual
            nonEliteId = (int) ((((BRKGA_pop_size - 1) - ELITE_SIZE) * curand_uniform(&state)) + ELITE_SIZE);
        }
        __syncthreads();
        int isElite = (curand_uniform(&state) < rhoe);
        nextPop->m_chromossomes[individualId].m_chromossome[aleloId] = isElite ?
                                                        prevPop->m_chromossomes[eliteId].m_chromossome[aleloId] :
                                                        prevPop->m_chromossomes[nonEliteId].m_chromossome[aleloId];
        nextPop->m_chromossomes[individualId].m_fitness = 0;
    }
    globalState[tid] = state;
}


///////////////////////////////// CUDA FUNCTIONS - END ///////////////////////////////////////////////////////


void loadFromFile(std::vector<string>& instanceData) {
    //COLOQUE AQUI O ARQUIVO QUE CONTÉM A INSTÂNCIA QUE PRETENDE EXECUTAR
    string instance_dir = "/home/lapo/mfonseca/Documents/Patricia/Instance/20caracteres/";

    //ARQUIVOS DE INSTÂNCIAS A SEREM UTILIZADOS
    std::vector<string> instance_name;
    instance_name.push_back("t20-10-500-1");
    //instance_name.push_back("n10m500tai1");
    // carregar o arquvio com as instâncias
    string tmp_name(instance_dir + instance_name[0] + ".txt");
    char * filename = new char[tmp_name.size() + 1];
    strcpy(filename, tmp_name.c_str());

    std::vector<int> max, min, avg;

    ifstream fin(filename);

    int dummy;
    fin >> dummy;
    assert(dummy == INSTANCE_SIZE);
    fin >> dummy;
    assert(dummy == CHROMOSSOME_SIZE);

    string tmp = "";

    for(int i = 0; i < INSTANCE_SIZE; ++i) {
        fin >> tmp;
        instanceData.push_back(tmp);
        tmp = "";
    }
    fin.close();
    delete [] filename;
}

void printInitialPopulation(Population * pop) {
    for (int i = 0; i < BRKGA_pop_size; ++i) {
        printf("Chromossome %d: ", i);
        for (int j = 0; j< CHROMOSSOME_SIZE; ++j) {
            printf("%.2f, ", pop->m_chromossomes[i].m_chromossome[j]);
        }
        printf("\n");
    }
}

void printSortedPopulation(Population * pop) {
    for (int i = 0; i < BRKGA_pop_size; ++i) {
        printf("Chromossome: %d - Fitness: %d\n", i, pop->m_chromossomes[i].m_fitness);
    }
}

float average(Population * pop){
    int sum = 0;
    for (int i = 0; i < BRKGA_pop_size; ++i) {
        sum += pop->m_chromossomes[i].m_fitness;
    }
    return sum/BRKGA_pop_size;
}

int main(int argc, char **argv) {
    cudaEvent_t start, stop, randomTime;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&randomTime);
    cudaEventRecord(start, 0);

    std::vector<string> tmpInstanceData;
    loadFromFile(tmpInstanceData);
    assert(tmpInstanceData.size() == INSTANCE_SIZE);

    //char h_alphabet[ALPHABET_SIZE +1];
    //strcpy(h_alphabet, "0123456789abcdefghij");

    // convert string vector into char
    char instanceData[INSTANCE_SIZE][CHROMOSSOME_SIZE + 1];
    for (int i = 0; i < INSTANCE_SIZE ; ++i) {
        strcpy(instanceData[i], tmpInstanceData[i].c_str());
    }

    // CPU variables:
    Population * h_pop;
    Instance * h_instance;
    h_pop = (struct Population *)calloc(1,sizeof(struct Population));
    h_instance = (struct Instance *)calloc(1,sizeof(struct Instance));
    //memcpy(h_instance->m_alphabet, h_alphabet, ALPHABET_SIZE+1);
    memcpy(h_instance->m_instanceData, instanceData, INSTANCE_SIZE * (CHROMOSSOME_SIZE + 1) * sizeof(char));

    // GPU variables
    Population * d_pop, *d_popPrevious;
    curandState* devStates;
    Instance * d_instance;

    //dim3 nblocks(NUM_BLOCKS,INSTANCE_SIZE);

    CUDA_CALL(cudaMalloc((void **)&devStates, (NUM_BLOCKS * NUM_THREADS) *sizeof(curandState)));
    CUDA_CALL(cudaMalloc((void **)&d_pop, sizeof(Population)));
    CUDA_CALL(cudaMalloc((void **)&d_popPrevious, sizeof(Population)));
    CUDA_CALL(cudaMalloc((void **)&d_instance, sizeof(Instance)));

    // copying CPU values to GPU values
    CUDA_CALL(cudaMemcpy(d_instance, h_instance, sizeof(Instance), cudaMemcpyHostToDevice));

    // Generate Initial Population
    setupRand <<<NUM_BLOCKS, NUM_THREADS >>> (devStates, time(NULL));
    cudaEventRecord(randomTime, 0);
    cudaEventSynchronize(randomTime);
    cudaEventElapsedTime(&elapsedTime, start, randomTime);
    std::cout << "Time to generate random value: " << (elapsedTime/1000) << std::endl;
    genInitialPop <<<NUM_BLOCKS, NUM_THREADS>>> (devStates, d_pop);
    //time_t TStop = time(NULL), TStart = time(NULL);
    int i = 0;
    while (true) {
        evaluate <<<NUM_BLOCKS, NUM_THREADS>>>(d_pop, d_instance);
        // Sort chromossomes by fitness
        thrust::device_ptr<Individual> t_chromossomes(d_pop->m_chromossomes);
        thrust::sort(t_chromossomes, t_chromossomes + BRKGA_pop_size);
        //CUDA_CALL(cudaMemcpy(h_pop, d_pop, sizeof(Population), cudaMemcpyDeviceToHost));
        //cout << "iteration " << i << ": " << h_pop->m_chromossomes[0].m_fitness << std::endl;
        //float avg = average(h_pop);
        // If end criteria was reached end execution
        //if ((TStop - TStart) > EXECUTION_TIME) break;
        if (i >= 500) break;
        // Generate next Population
        // First copy the current pop to previous:
        CUDA_CALL(cudaMemcpy(d_popPrevious, d_pop, sizeof(Population), cudaMemcpyDeviceToDevice));
        // call kernel to generate new population
        generateNextPop <<<NUM_BLOCKS, NUM_THREADS>>>(devStates, d_popPrevious, d_pop);
        //TStop = time(NULL);
        ++i;
    }

    CUDA_CALL(cudaMemcpy(h_pop, d_pop, sizeof(Population), cudaMemcpyDeviceToHost));
    std::cout << h_pop->m_chromossomes[0].m_fitness;
    //DEBUG_BRKGA(printSortedPopulation(h_pop);)
    //float avg = average(h_pop);
    //printf("Best chromossome: %d - Average: %.2f - Worst chromossome: %d\n",h_pop->m_chromossomes[0].m_fitness, avg, h_pop->m_chromossomes[BRKGA_pop_size-1].m_fitness);
    //printf("\n");
    // Free GPU memory
    CUDA_CALL(cudaFree(d_pop));
    CUDA_CALL(cudaFree(d_popPrevious));
    CUDA_CALL(cudaFree(devStates));
    CUDA_CALL(cudaFree(d_instance));
    // Free CPU memory
    free (h_pop);
    free (h_instance);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "   " << (elapsedTime/1000) << std::endl;
    //printf ("Your calculations took %.15lf seconds to run.\n", dif );
    return 0;
}
