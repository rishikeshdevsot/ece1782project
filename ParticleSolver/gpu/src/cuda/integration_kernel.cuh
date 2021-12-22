#ifndef INTEGRATION_KERNEL_H
#define INTEGRATION_KERNEL_H

#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <thrust/sort.h>

#include "helper_math.h"
#include "math_constants.h"
#include "kernel.cuh"
#include "shared_variables.cuh"

//#define X_BOUNDARY 7.f
//#define X_BOUNDARY 50.f
//#define Z_BOUNDARY 50.f

//#define

#define EPS 0.001f

////////////// fluid constants /////////////
#define MAX_FLUID_NEIGHBORS 150

#define H 2.f       // kernel radius
#define H2 4.f      // H^2
#define H6 64.f     // H^6
#define H9 512.f    // H^9
#define POLY6_COEFF 0.00305992474f // 315 / (64 * pi * H9)
#define SPIKEY_COEFF 0.22381163872f // 45 / (pi * H6)

#define FLUID_RELAXATION .01f // epsilon used when calculating lambda
#define K_P .1f              // scales artificial pressure
#define E_P 4.f              // exponent to art. pressure
#define DQ_P .2f             // between .1 and .3 (for art pressure)


/////////////////// friction ///////////////
#define S_FRICTION .005f
#define K_FRICTION .0002f
//#define S_FRICTION .15f
//#define K_FRICTION .003f

// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float, 1, cudaReadModeElementType> invMassTex;
texture<int, 1, cudaReadModeElementType> oldPhaseTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;


// simulation parameters in constant memory
__constant__ SimParams params;
SimParams h_params;

struct collide_world_functor
{
    float *rands;
    int3 minBounds;
    int3 maxBounds;

    __host__ __device__
    collide_world_functor(float *_rands, int3 _minBounds, int3 _maxBounds)
        : rands(_rands), minBounds(_minBounds), maxBounds(_maxBounds) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        float4 posData = thrust::get<0>(t);
        float4 Xstar = thrust::get<1>(t);
        int phase = thrust::get<2>(t);

        float3 epos = make_float3(posData.x, posData.y, posData.z);
        float3 pos = make_float3(Xstar.x, Xstar.y, Xstar.z);

        float3 n = make_float3(0.f);

        float d = params.particleRadius;
        float eps = d * 0.f;
        if (phase < SOLID)
            eps = d * 0.01f;

        if (epos.y < minBounds.y + params.particleRadius)
        {
            epos.y = minBounds.y + params.particleRadius + rands[5] * eps;
            n += make_float3(0,1,0);
        }

        eps = d * 0.01f;

        if (epos.x > maxBounds.x - params.particleRadius)
        {
            epos.x = maxBounds.x - (params.particleRadius + rands[0] * eps);
            n += make_float3(-1,0,0);
        }

        if (epos.x < minBounds.x + params.particleRadius)
        {
            epos.x = minBounds.x + (params.particleRadius + rands[1] * eps);
            n += make_float3(1,0,0);
        }

        if (epos.y > maxBounds.y - params.particleRadius)
        {
            epos.y = maxBounds.y - (params.particleRadius + rands[2] * eps);
            n += make_float3(0,-1,0);
        }

#ifndef TWOD
        if (epos.z > maxBounds.z - params.particleRadius)
        {
            epos.z = maxBounds.z - (params.particleRadius + rands[3] * eps);
            n += make_float3(0,0,-1);
        }

        if (epos.z < minBounds.z + params.particleRadius)
        {
            epos.z = minBounds.z + (params.particleRadius + rands[4] * eps);
            n += make_float3(0,0,1);
        }
#endif


#ifdef TWOD
        epos.z = ZPOS; // 2D
        pos.z = ZPOS;
#endif

        if (length(n) < EPS || phase < CLOTH)
        {
            thrust::get<0>(t) = make_float4(epos, posData.w);
            return;
        }

        float3 dp = (epos - pos);
        float3 dpt = dp - dot(dp, n) * n;
        float ldpt = length(dpt);

        if (ldpt < EPS)
        {
            thrust::get<0>(t) = make_float4(epos, posData.w);
            return;
        }


        if (ldpt < sqrt(S_FRICTION) * d)
            epos -= dpt;
        else
            epos -= dpt * min(sqrt(K_FRICTION) * d / ldpt, 1.f);

        // store new position and velocity

        thrust::get<0>(t) = make_float4(epos, posData.w);
    }
};

struct integrate_functor
{
    float deltaTime;

    __host__ __device__
    integrate_functor(float delta_time)
        : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);

        vel += params.gravity * deltaTime;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
    }
};

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

int3 calcGridPos_cpu(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - h_params.worldOrigin.x) / h_params.cellSize.x);
    gridPos.y = floor((p.y - h_params.worldOrigin.y) / h_params.cellSize.y);
    gridPos.z = floor((p.z - h_params.worldOrigin.z) / h_params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y-1);
    gridPos.z = gridPos.z & (params.gridSize.z-1);
    return ((gridPos.z * params.gridSize.y) * params.gridSize.x) + gridPos.y * params.gridSize.x + gridPos.x;
}

uint calcGridHash_cpu(int3 gridPos)
{
    gridPos.x = gridPos.x & (h_params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (h_params.gridSize.y-1);
    gridPos.z = gridPos.z & (h_params.gridSize.z-1);
    return ((gridPos.z * h_params.gridSize.y) * h_params.gridSize.x) + (gridPos.y * h_params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint   *gridParticleHash,  // output
               uint   *gridParticleIndex, // output
               float4 *pos,               // input: positions
               uint    numParticles)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  float4 *sortedPos,        // output: sorted positions
                                  float  *sortedW,          // output: sorted inverse masses
                                  int    *sortedPhase,      // output: sorted phase values
                                  uint   *gridParticleHash, // input: sorted grid hashes
                                  uint   *gridParticleIndex,// input: sorted particle indices
                                  float4 *oldPos,           // input: position array
                                  float  *W,
                                  int    *phase,
                                  uint    numParticles)
{
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }

    __syncthreads();

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = gridParticleIndex[index];
        float4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
        float w = FETCH(invMass, sortedIndex);       // macro does either global read or texture fetch
        int phase = FETCH(oldPhase, sortedIndex);       // macro does either global read or texture fetch

        sortedPos[index] = pos;
        sortedW[index] = w;
        sortedPhase[index] = phase;
    }


}


// collide a particle against all other particles in a given cell
__device__
void collideCell(int3    gridPos,
                 uint    index,
                 float3  pos,
                 int     phase,
                 float4 *oldPos,
                 uint   *cellStart,
                 uint   *cellEnd,
                 uint   *neighbors,
                 uint   *numNeighbors)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, gridHash);

    float collideDist = params.particleRadius * 2.001f; // slightly bigger radius
    float collideDist2 = collideDist * collideDist;

//    float3 delta = make_float3(0.0f);

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, gridHash);

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {
                float3 pos2 = make_float3(FETCH(oldPos, j));
                int phase2 = FETCH(oldPhase, j);

                if (phase > SOLID && phase == phase2)
                    continue;

                // collide two spheres
                float3 diff = pos - pos2;

                float mag2 = dot(diff, diff);

                if (mag2 < collideDist2 && numNeighbors[index] < MAX_FLUID_NEIGHBORS)
                {
                    // neighbor stuff
                    neighbors[index * MAX_FLUID_NEIGHBORS + numNeighbors[index]] = j;
                    numNeighbors[index] += 1;

//                    delta += diff * (sqrt(mag2) - collideDist) * -.5f;
                }
            }
        }
    }
}


__global__
void collideD(float4 *newPos,               // output: new pos
              float4 *prevPositions,
              float4 *sortedPos,               // input: sorted positions
              float  *sortedW,
              int    *sortedPhase,
              uint   *gridParticleIndex,    // input: sorted particle indices
              uint   *cellStart,
              uint   *cellEnd,
              uint    numParticles,
              uint   *neighbors,
              uint   *numNeighbors)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numParticles) return;

    int phase = FETCH(oldPhase, index);
    if (phase < CLOTH) return;

    // read particle data from sorted arrays
    float3 pos = make_float3(FETCH(oldPos, index));

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    float3 delta = make_float3(0.f);

    numNeighbors[index] = 0;
    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                collideCell(neighbourPos, index, pos, phase, sortedPos, cellStart, cellEnd, neighbors, numNeighbors);
            }
        }
    }

    float collideDist = params.particleRadius * 2.001f;

    float w = FETCH(invMass, index);
    float sW = (w != 0.f ? (1.f / ((1.f / w) * exp(-pos.y))) : w);

    uint originalIndex = gridParticleIndex[index];
//    float3 currPos = make_float3(newPos[originalIndex]);
    float3 prevPos = make_float3(prevPositions[originalIndex]);

    for (uint i = 0; i < numNeighbors[index]; i++)
    {
        float3 pos2 =  make_float3(FETCH(oldPos, neighbors[index * MAX_FLUID_NEIGHBORS + i]));
        float w2 =  FETCH(invMass, neighbors[index * MAX_FLUID_NEIGHBORS + i]);
        int phase2 =  FETCH(oldPhase, neighbors[index * MAX_FLUID_NEIGHBORS + i]);

        float3 diff = pos - pos2;
        float dist = length(diff);
        float mag = dist - collideDist;

        float colW = w;
        float colW2 = w2;

        if (phase >= SOLID && phase2 >= SOLID)
        {
            colW = sW;
            colW2 = (w2 != 0.f ? (1.f / ((1.f / w2) * exp(-pos.y))) : w2);
        }

//        colWsum = colW + colW1);
        float scale = mag / (colW + colW2);
        float3 dp = diff * (scale / dist);
        float3 dp1 = -colW * dp / numNeighbors[index];
        float3 dp2 = colW2 * dp / numNeighbors[index];

        delta += dp1;



        ////////////////////// friction //////////////////
        if (phase < SOLID || phase2 < SOLID)
            continue;

        uint neighborIndex = gridParticleIndex[neighbors[index * MAX_FLUID_NEIGHBORS + i]];
        float3 prevPos2 = make_float3(prevPositions[neighborIndex]);
//        float3 currPos2 = make_float3(newPos[neighbors[index * MAX_FLUID_NEIGHBORS + i]]);

        float3 nf = normalize(diff);
        float3 dpRel = (pos + dp1 - prevPos) - (prevPos + dp2 - prevPos2);
        float3 dpt = dpRel - dot(dpRel, nf) * nf;
        float ldpt = length(dpt);

        if (ldpt < EPS)
            continue;

        if (ldpt < (S_FRICTION) * dist)
            delta -= dpt * colW / (colW + colW2);
        else
            delta -= dpt * min((K_FRICTION) * dist / ldpt, 1.f);
    }

    // write new velocity back to original unsorted location
    newPos[originalIndex] = make_float4(pos + delta, 1.0f);
}


struct subtract_functor
{
    const float time;

    subtract_functor(float _time) : time(_time) {}

    __device__
    float4 operator()(const float4& orig, const float4& solved) const {
        return (solved - orig) / -time;
    }
};



//TODO NEXT: oldpos of neighbors is loaded 3 times, load once is enough, just make them 
// to shared memory and reuse for the later 2 accesses.
#define RADHARDCODE (3)
__global__
void SolveFluidsFused(float  *lambda,               // input: sorted positions
                  uint   *gridParticleIndex,    // input: sorted particle indices
//                  uint   *cellStart,
//                  uint   *cellEnd,
                  uint    numParticles,
                  uint   *neighbors,
                  uint   *numNeighbors,
                  float4 *particles,
                  float  *ros)
                  // float4 *neighborsPosCache)
{
    //__shared__ uint neighborPosCache[64 * MAX_FLUID_NEIGHBORS];

    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;

    int phase = FETCH(oldPhase, index);
    if (phase != FLUID) return;

    // read particle data from sorted arrays
    float4 k2_pos = FETCH(oldPos, index);
    float3 pos = make_float3(k2_pos);

    float w = FETCH(invMass, index);

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    // TODO: make constant
    int rad = (int)ceil(H / params.cellSize.x);
    //printf("rad: %d\n",rad);

    // TODO: eliminate the access of this into just a single write, everything else is inrelavant here
    unsigned int numNeighborsLocal = 0;

    // #pragma unroll
    for (int z=-RADHARDCODE; z<=RADHARDCODE; z++)
    {
    //    #pragma unroll
        for (int y=-RADHARDCODE; y<=RADHARDCODE; y++)
        {
    //        #pragma unroll
            for (int x=-RADHARDCODE; x<=RADHARDCODE; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                uint gridHash = calcGridHash(neighbourPos);

                // get start of bucket for this cell
                uint startIndex = FETCH(cellStart, gridHash);

                if (startIndex != 0xffffffff)          // cell is not empty
                {
                    // iterate over particles in this cell
                    uint endIndex = FETCH(cellEnd, gridHash);
                    //printf("number of particles per grid: %d\n", endIndex - startIndex);
                    for (uint j=startIndex; j<endIndex; j++)
                    {
                        if (j != index)                // check not colliding with self
                        {
                            // TODO: pos2 can be saved into shared memory

                            float4 pos2_f4 = FETCH(oldPos, j);
                            float3 pos2 = make_float3(pos2_f4);

                            float3 relPos = pos - pos2;
                            float dist2 = dot(relPos, relPos);
                            if (dist2 < H2 && numNeighborsLocal < MAX_FLUID_NEIGHBORS)
                            {
                                // neighbor stuff
                                // TODO: coalse the wirte to this variable
                                //neighborPosCache[threadIdx.x * MAX_FLUID_NEIGHBORS + numNeighborsLocal] = j;
                                neighbors[index * MAX_FLUID_NEIGHBORS + numNeighborsLocal] = j;
                                // neighborsPosCache[index * MAX_FLUID_NEIGHBORS + numNeighborsLocal] = pos2_f4;
                                numNeighborsLocal += 1;
                            }
                        }
                    }
                }

            }
        }
    }
    // we do not need this because now kernels are fused
    // numNeighbors[index] = numNeighborsLocal;
    //printf("num neighbors: %d\n", numNeighborsLocal);


    float ro = 0.f;
    float denom = 0.f;
    float3 grad = make_float3(0.f);
    uint gridParticleIndexLocal = gridParticleIndex[index];
    float rosLocal = ros[gridParticleIndexLocal];
    for (uint i = 0; i < numNeighborsLocal; i++)
    {
        // TODO: this again is a global memory read that we do not need so far
        float3 pos2 =  make_float3(FETCH(oldPos, neighbors[index * MAX_FLUID_NEIGHBORS + i]));
        //float3 pos2 =  make_float3(FETCH(oldPos, neighborPosCache[threadIdx.x * MAX_FLUID_NEIGHBORS + i]));
//      float w2 = FETCH(invMass, ni);
        float3 r = pos - pos2;
        float rlen2 = dot(r, r);
        float rlen = sqrt(rlen2);
        float hMinus2 = H2 - rlen2;
        float hMinus = H - rlen;

        // do fluid solid scaling hurr
        ro += (POLY6_COEFF * hMinus2*hMinus2*hMinus2 ) / w;

        float3 spikeyGrad;
        if (rlen < 0.0001f)
            spikeyGrad = make_float3(0.f); // randomize a little
        else
            spikeyGrad = (r / rlen) * -SPIKEY_COEFF * hMinus*hMinus;
        spikeyGrad /= rosLocal;

        grad += -spikeyGrad;
        denom += dot(spikeyGrad, spikeyGrad);
    }
    // make this constexpr
    ro += (POLY6_COEFF * H6 ) / w;
    denom += dot(grad, grad);
    // TODO: this definitely can go shared memory if we fuse the 2 kernels
    float lamdaLocal = - ((ro / rosLocal) - 1) / (denom + FLUID_RELAXATION);
    lambda[index] = lamdaLocal;
    ///////////////////////////// THIS IS THE KERNEL FUSION BOUNDARY ///////////////////////
    __syncthreads();


    float4 delta = make_float4(0.f);
    for (uint i = 0; i < numNeighborsLocal; i++)
    {
        float4 k2_pos2 = FETCH(oldPos, neighbors[index * MAX_FLUID_NEIGHBORS + i]);
        //float4 k2_pos2 = FETCH(oldPos, neighborPosCache[threadIdx.x * MAX_FLUID_NEIGHBORS + i]);

        float4 r = k2_pos - k2_pos2;
        float rlen2 = dot(r, r);
        float rlen = sqrt(rlen2);
        float hMinus2 = H2 - rlen2;
        float hMinus = H - rlen;

        float4 spikeyGrad;
        if (rlen < 0.0001f)
            spikeyGrad = make_float4(0,EPS,0,0) * -SPIKEY_COEFF * hMinus*hMinus;
        else
            spikeyGrad = (r / rlen) * -SPIKEY_COEFF * hMinus*hMinus;

        float term2 = H2 - (DQ_P * DQ_P * H2);

        float numer = (POLY6_COEFF * hMinus2*hMinus2*hMinus2 ) ;
        float denom = (POLY6_COEFF * term2*term2*term2 );
        float lambdaCorr = -K_P * pow(numer / denom, E_P);

        delta += (lamdaLocal + lambda[neighbors[index * MAX_FLUID_NEIGHBORS + i]] + lambdaCorr) * spikeyGrad;
        // delta += (lambda[index] + lambda[neighborPosCache[threadIdx.x * MAX_FLUID_NEIGHBORS + i]] + lambdaCorr) * spikeyGrad;
    }

    particles[gridParticleIndexLocal] += delta / (rosLocal + numNeighborsLocal);
}


// collide a particle against all other particles in a given cell
__device__
void collideCellRadius(int3    gridPos,
                       uint    index,
                       float3  pos,
                       uint   *cellStart,
                       uint   *cellEnd,
                       uint   *neighbors,
                       uint   *numNeighbors)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, gridHash);

    int num_neighbors = numNeighbors[index];
    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, gridHash);

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {
                // TODO: pos2 can be saved into shared memory
                float3 pos2 = make_float3(FETCH(oldPos, j));

                float3 relPos = pos - pos2;
                float dist2 = dot(relPos, relPos);
                if (dist2 < H2 && num_neighbors < MAX_FLUID_NEIGHBORS)
                {
                    // neighbor stuff
                    // TODO: coalse the wirte to this variable
                    neighbors[index * MAX_FLUID_NEIGHBORS + num_neighbors] = j;
                    num_neighbors += 1;
                }
            }
        }
    }
    numNeighbors[index] = num_neighbors;
}

void collideCellRadius_cpu(int3    gridPos,
                         uint    index,
                         float3  pos,
                         uint   *cellStart,
                         uint   *cellEnd,
                         float4   *oldPos,
                         uint   *neighbors,
                         uint   *numNeighbors)
{
    uint gridHash = calcGridHash_cpu(gridPos);

    // get start of bucket for this cell
    uint startIndex = cellStart[gridHash];

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {
                float3 pos2 = make_float3(oldPos[j]);

                float3 relPos = pos - pos2;
                float dist2 = dot(relPos, relPos);
                if (dist2 < H2 && numNeighbors[index] < MAX_FLUID_NEIGHBORS)
                {
                    // neighbor stuff
                    neighbors[index * MAX_FLUID_NEIGHBORS + numNeighbors[index]] = j;
                    numNeighbors[index] += 1;
                }
            }
        }
    }

}


__global__
void findLambdasD(float  *lambda,               // input: sorted positions
                  uint   *gridParticleIndex,    // input: sorted particle indices
                  uint   *cellStart,
                  uint   *cellEnd,
                  uint    numParticles,
                  uint   *neighbors,
                  uint   *numNeighbors,
                  float  *ros)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    int phase = FETCH(oldPhase, index);
    if (phase != FLUID) return;

    // read particle data from sorted arrays
    float3 pos = make_float3(FETCH(oldPos, index));

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells

    int rad = (int)ceil(H / params.cellSize.x);

    numNeighbors[index] = 0;
    for (int z=-rad; z<=rad; z++)
    {
        for (int y=-rad; y<=rad; y++)
        {
            for (int x=-rad; x<=rad; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                collideCellRadius(neighbourPos, index, pos, cellStart, cellEnd, neighbors, numNeighbors);
            }
        }
    }

    float w = FETCH(invMass, index);
    float ro = 0.f;
    float denom = 0.f;
    float3 grad = make_float3(0.f);
    for (uint i = 0; i < numNeighbors[index]; i++)
    {
        uint ni = neighbors[index * MAX_FLUID_NEIGHBORS + i];
        float3 pos2 =  make_float3(FETCH(oldPos, ni));
//        float w2 = FETCH(invMass, ni);
        float3 r = pos - pos2;
        float rlen2 = dot(r, r);
        float rlen = sqrt(rlen2);
        float hMinus2 = H2 - rlen2;
        float hMinus = H - rlen;

        // do fluid solid scaling hurr
        ro += (POLY6_COEFF * hMinus2*hMinus2*hMinus2 ) / w;

        float3 spikeyGrad;
        if (rlen < 0.0001f)
            spikeyGrad = make_float3(0.f); // randomize a little
        else
            spikeyGrad = (r / rlen) * -SPIKEY_COEFF * hMinus*hMinus;
        spikeyGrad /= ros[gridParticleIndex[index]];

        grad += -spikeyGrad;
        denom += dot(spikeyGrad, spikeyGrad);
    }
    ro += (POLY6_COEFF * H6 ) / w;
    denom += dot(grad, grad);

    lambda[index] = - ((ro / ros[gridParticleIndex[index]]) - 1) / (denom + FLUID_RELAXATION);
}

__global__ void 
__launch_bounds__(32 /*maxThreadsPerBlock */, 16/*minBlocksPerMultiprocessor */)
findLambdasDOptimized(float  *lambda,               // input: sorted positions
                           uint   *gridParticleIndex,    // input: sorted particle indices
                           float  *oldPos,
                           uint   *cellStart,
                           uint   *cellEnd,
                           uint    numParticles,
                           uint   *neighbors,
                           uint   *numNeighbors,
                           float  *ros,
                           size_t cache_size)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numParticles) return;

    int phase = FETCH(oldPhase, index);
    if (phase != FLUID) return;

    extern __shared__ char cache[];
    // max particles is 50000 so short (65532) will not ovetrflow
    unsigned short *neighbor_cache = (unsigned short*)cache;
    size_t cache_cap = cache_size / 2; // short is 2 bytes
    size_t cache_per_thread = cache_cap / blockDim.x;

    // read particle data from sorted arrays
    float3 pos = make_float3(oldPos[index*4],oldPos[index*4+1],oldPos[index*4+2]);

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells

    int num_neighbors = 0;
    for (int z=-RADHARDCODE; z<=RADHARDCODE; z++)
    {
        for (int y=-RADHARDCODE; y<=RADHARDCODE; y++)
        {
            int3 neighbourPos[RADHARDCODE*2+1];
            uint gridHash[RADHARDCODE*2+1];
            uint startIndex[RADHARDCODE*2+1];
#pragma unroll
            for (int x=-RADHARDCODE; x<=RADHARDCODE; x++)
            {
                neighbourPos[x+RADHARDCODE] = gridPos + make_int3(x, y, z);
                gridHash[x+RADHARDCODE] = calcGridHash(neighbourPos[x+RADHARDCODE]);

                // get start of bucket for this cell
                startIndex[x+RADHARDCODE] = cellStart[gridHash[x+RADHARDCODE]];
            }

//#pragma unroll
            for (int x=-RADHARDCODE; x<=RADHARDCODE; x++)
            {
                if (startIndex[x+RADHARDCODE] != 0xffffffff)          // cell is not empty
                {
                    // iterate over particles in this cell
                    uint endIndex = cellEnd[gridHash[x+RADHARDCODE]];

                    for (uint j=startIndex[x+RADHARDCODE]; j<endIndex; j++)
                    {
                        if (j != index)                // check not colliding with self
                        {
                            // TODO: pos2 can be saved into shared memory
                            float3 pos2 = make_float3(oldPos[j*4],oldPos[j*4+1],oldPos[j*4+2]);

                            float3 relPos = pos - pos2;
                            float dist2 = dot(relPos, relPos);
                            if (dist2 < H2 && num_neighbors < MAX_FLUID_NEIGHBORS)
                            {
                                // neighbor stuff
                                // TODO: coalse the wirte to this variable
                                if (num_neighbors < cache_per_thread) {
                                    neighbor_cache[threadIdx.x * cache_per_thread + num_neighbors] = j;    
                                }
                                else {
                                    neighbors[index * MAX_FLUID_NEIGHBORS + num_neighbors] = j;
                                }
                                num_neighbors += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    numNeighbors[index] = num_neighbors;

    float w = FETCH(invMass, index);
    float ro = 0.f;
    float denom = 0.f;
    float3 grad = make_float3(0.f);
    uint gridParticleIndexLocal = gridParticleIndex[index];
    float rosLocal = ros[gridParticleIndexLocal];

    //  read from cache
    for (uint i = 0; i < cache_per_thread; i++)
    {
        if (i >= num_neighbors) break;

        uint ni = neighbor_cache[threadIdx.x * cache_per_thread + i];
        float3 pos2 =  make_float3(oldPos[ni*4],oldPos[ni*4+1],oldPos[ni*4+2]);
        float3 r = pos - pos2;
        float rlen2 = dot(r, r);
        float rlen = sqrt(rlen2);
        float hMinus2 = H2 - rlen2;
        float hMinus = H - rlen;

        // do fluid solid scaling hurr
        ro += (POLY6_COEFF * hMinus2*hMinus2*hMinus2 ) / w;

        float3 spikeyGrad;
        if (rlen < 0.0001f)
            spikeyGrad = make_float3(0.f); // randomize a little
        else
            spikeyGrad = (r / rlen) * -SPIKEY_COEFF * hMinus*hMinus;
        spikeyGrad /= rosLocal;

        grad += -spikeyGrad;
        denom += dot(spikeyGrad, spikeyGrad);
    }
    // read from the rest
    for (uint i = cache_per_thread; i < num_neighbors; i++)
    {
        uint ni = neighbors[index * MAX_FLUID_NEIGHBORS + i];
        float3 pos2 =  make_float3(oldPos[ni*4],oldPos[ni*4+1],oldPos[ni*4+2]);
        float3 r = pos - pos2;
        float rlen2 = dot(r, r);
        float rlen = sqrt(rlen2);
        float hMinus2 = H2 - rlen2;
        float hMinus = H - rlen;

        // do fluid solid scaling hurr
        ro += (POLY6_COEFF * hMinus2*hMinus2*hMinus2 ) / w;

        float3 spikeyGrad;
        if (rlen < 0.0001f)
            spikeyGrad = make_float3(0.f); // randomize a little
        else
            spikeyGrad = (r / rlen) * -SPIKEY_COEFF * hMinus*hMinus;
        spikeyGrad /= rosLocal;

        grad += -spikeyGrad;
        denom += dot(spikeyGrad, spikeyGrad);
    }
    ro += (POLY6_COEFF * H6 ) / w;
    denom += dot(grad, grad);

    lambda[index] = - ((ro / rosLocal) - 1) / (denom + FLUID_RELAXATION);

    // write cache back
    for (uint i = 0; i < cache_per_thread; i++)
    {
        if (i >= num_neighbors) break;

         neighbors[index * MAX_FLUID_NEIGHBORS + i] = \
                     neighbor_cache[threadIdx.x * cache_per_thread + i];
    }
}

void findLambdasD_cpu(float  *lambda,               // input: sorted positions
                  uint   *gridParticleIndex,    // input: sorted particle indices
                  uint   *cellStart,
                  uint   *cellEnd,
                  float4  *oldPos,
                  float   *invMass,
                  int   *oldPhase,
                  uint    numParticles,
                  uint   *neighbors,
                  uint   *numNeighbors,
                  float  *ros)
{
#pragma omp parallel for
    for(uint index = 0; index < numParticles; index = index + 1){
       // uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;


        int phase = oldPhase[index];
        if (phase != FLUID) continue;

        // read particle data from sorted arrays
        float3 pos = make_float3(oldPos[index]);

        // get address in grid 
        int3 gridPos = calcGridPos_cpu(pos);

        // examine neighbouring cells

        int rad = (int)ceil(H / h_params.cellSize.x);

        numNeighbors[index] = 0;
        for (int z=-rad; z<=rad; z++)
        {
            for (int y=-rad; y<=rad; y++)
            {
                for (int x=-rad; x<=rad; x++)
                {
                    int3 neighbourPos = gridPos + make_int3(x, y, z);
                    collideCellRadius_cpu(neighbourPos, index, pos, cellStart, cellEnd, oldPos, neighbors, numNeighbors);
                }
            }
        }

        float w = invMass[index];
        float ro = 0.f;
        float denom = 0.f;
        float3 grad = make_float3(0.f);
        for (uint i = 0; i < numNeighbors[index]; i++)
        {
            uint ni = neighbors[index * MAX_FLUID_NEIGHBORS + i];
            float3 pos2 =  make_float3(oldPos[ni]);
    //        float w2 = FETCH(invMass, ni);
            float3 r = pos - pos2;
            float rlen2 = dot(r, r);
            float rlen = sqrt(rlen2);
            float hMinus2 = H2 - rlen2;
            float hMinus = H - rlen;

            // do fluid solid scaling hurr
            ro += (POLY6_COEFF * hMinus2*hMinus2*hMinus2 ) / w;

            float3 spikeyGrad;
            if (rlen < 0.0001f)
                spikeyGrad = make_float3(0.f); // randomize a little
            else
                spikeyGrad = (r / rlen) * -SPIKEY_COEFF * hMinus*hMinus;
            spikeyGrad /= ros[gridParticleIndex[index]];

            grad += -spikeyGrad;
            denom += dot(spikeyGrad, spikeyGrad);
        }
        ro += (POLY6_COEFF * H6 ) / w;
        denom += dot(grad, grad);

        lambda[index] = - ((ro / ros[gridParticleIndex[index]]) - 1) / (denom + FLUID_RELAXATION);
    }
}

__global__ void 
__launch_bounds__(32 /*maxThreadsPerBlock */, 16/*minBlocksPerMultiprocessor */)
solveFluidsDOptimized(float  *lambda,              // input: sorted positions
                           uint   *gridParticleIndex,    // input: sorted particle indices
                           float  *oldPos,
                           float  *particles,
                           uint    numParticles,
                           uint   *neighbors,
                           uint   *numNeighbors,
                           float  *ros)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numParticles) return;

    int phase = FETCH(oldPhase, index);
    if (phase != FLUID) return;

    float3 pos = make_float3(oldPos[index*4],oldPos[index*4+1],oldPos[index*4+2]);

    float3 delta = make_float3(0.f);
    uint numNeighborsLocal = numNeighbors[index];
    uint gridParticleIndexLocal = gridParticleIndex[index];
    float lamdaLocal = lambda[index];

    for (uint i = 0; i < numNeighborsLocal; i++)
    {
        uint neighborsLocal = neighbors[index * MAX_FLUID_NEIGHBORS + i];
        float3 pos2 =  make_float3(oldPos[neighborsLocal*4], oldPos[neighborsLocal*4+1], oldPos[neighborsLocal*4+2]);
        float3 r = pos - pos2;
        float rlen2 = dot(r, r);
        float rlen = sqrt(rlen2);
        float hMinus2 = H2 - rlen2;
        float hMinus = H - rlen;

        float3 spikeyGrad;
        if (rlen < 0.0001f)
            spikeyGrad = make_float3(0,EPS,0) * -SPIKEY_COEFF * hMinus*hMinus;
        else
            spikeyGrad = (r / rlen) * -SPIKEY_COEFF * hMinus*hMinus;

        float term2 = H2 - (DQ_P * DQ_P * H2);

        float numer = (POLY6_COEFF * hMinus2*hMinus2*hMinus2 ) ;
        float denom = (POLY6_COEFF * term2*term2*term2 );
        float lambdaCorr = -K_P * pow(numer / denom, E_P);

        delta += (lamdaLocal + lambda[neighborsLocal] + lambdaCorr) * spikeyGrad;
    }

    float rosPlusNumNeighboursLocal = (ros[gridParticleIndexLocal] + numNeighborsLocal);
    particles[gridParticleIndexLocal*4] += delta.x / rosPlusNumNeighboursLocal;
    particles[gridParticleIndexLocal*4+1] += delta.y / rosPlusNumNeighboursLocal;
    particles[gridParticleIndexLocal*4+2] += delta.z / rosPlusNumNeighboursLocal;
}

__global__
void solveFluidsD(float  *lambda,              // input: sorted positions
                  uint   *gridParticleIndex,    // input: sorted particle indices
                  float4 *particles,
                  uint    numParticles,
                  uint   *neighbors,
                  uint   *numNeighbors,
                  float  *ros)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    int phase = FETCH(oldPhase, index);
    if (phase != FLUID) return;

    float4 pos = FETCH(oldPos, index);

    float4 delta = make_float4(0.f);
    for (uint i = 0; i < numNeighbors[index]; i++)
    {
        float4 pos2 =  FETCH(oldPos, neighbors[index * MAX_FLUID_NEIGHBORS + i]);
        float4 r = pos - pos2;
        float rlen2 = dot(r, r);
        float rlen = sqrt(rlen2);
        float hMinus2 = H2 - rlen2;
        float hMinus = H - rlen;

        float4 spikeyGrad;
        if (rlen < 0.0001f)
            spikeyGrad = make_float4(0,EPS,0,0) * -SPIKEY_COEFF * hMinus*hMinus;
        else
            spikeyGrad = (r / rlen) * -SPIKEY_COEFF * hMinus*hMinus;

        float term2 = H2 - (DQ_P * DQ_P * H2);

        float numer = (POLY6_COEFF * hMinus2*hMinus2*hMinus2 ) ;
        float denom = (POLY6_COEFF * term2*term2*term2 );
        float lambdaCorr = -K_P * pow(numer / denom, E_P);

        delta += (lambda[index] + lambda[neighbors[index * MAX_FLUID_NEIGHBORS + i]] + lambdaCorr) * spikeyGrad;
    }

    uint origIndex = gridParticleIndex[index];
    particles[origIndex] += delta / (ros[gridParticleIndex[index]] + numNeighbors[index]);

}

#endif // INTEGRATION_KERNEL_H
