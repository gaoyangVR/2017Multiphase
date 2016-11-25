#include <cuda_runtime.h>    // includes cuda.h and cuda_runtime_api.h
#include "spray_k.cuh"
#include<helper_cuda.h>
#include<helper_math.h>
#include "utility.h"
#include "tables.h"

__constant__ FlipConstant dparam;
__constant__ int NX;
__constant__ int NY;
__constant__ int NZ;
__constant__ int NXMC;
__constant__ int NYMC;
__constant__ int NZMC;
texture<uint, 1, cudaReadModeElementType> edgeTex;
texture<uint, 1, cudaReadModeElementType> triTex;
texture<uint, 1, cudaReadModeElementType> numVertsTex;


void copyparamtoGPU(FlipConstant hparam)
{
	checkCudaErrors(cudaMemcpyToSymbol(dparam, &hparam, sizeof(FlipConstant)));
}

void copyNXNYNZtoGPU(int nx, int ny, int nz)
{
	checkCudaErrors(cudaMemcpyToSymbol(NX, &nx, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(NY, &ny, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(NZ, &nz, sizeof(int)));
}

void copyNXNYNZtoGPU_MC(int nx, int ny, int nz)
{
	checkCudaErrors(cudaMemcpyToSymbol(NXMC, &nx, sizeof(int))); 
	checkCudaErrors(cudaMemcpyToSymbol(NYMC, &ny, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(NZMC, &nz, sizeof(int)));
}

__device__ inline void getijk(int &i, int &j, int &k, int &idx)
{
	i = idx / (NZ*NY);
	j = idx / NZ%NY;
	k = idx%NZ;
}

__device__ inline void getijkfrompos(int &i, int &j, int &k, float3 pos)
{
	pos = (pos - dparam.gmin) / dparam.cellsize;
	i = (pos.x >= 0 && pos.x<NX) ? ((int)pos.x) : 0;
	j = (pos.y >= 0 && pos.y<NY) ? ((int)pos.y) : 0;
	k = (pos.z >= 0 && pos.z<NZ) ? ((int)pos.z) : 0;
}
__device__ inline void getijkfrompos(int &i, int &j, int &k, float3 pos, int w, int h, int d, float dx)
{
	pos = (pos - dparam.gmin) / dx;
	i = (pos.x >= 0 && pos.x<w) ? ((int)pos.x) : 0;
	j = (pos.y >= 0 && pos.y<h) ? ((int)pos.y) : 0;
	k = (pos.z >= 0 && pos.z<d) ? ((int)pos.z) : 0;
}

__device__ inline int getidx(int i, int j, int k)
{
	return (i*NZ*NY + j*NZ + k);
}

__device__ inline int getidx(int i, int j, int k, int w, int h, int d)
{
	return (i*h*d + j*d + k);
}

__device__ inline float getRfromMass(float m)
{
	return pow(m*0.75f / M_PI / dparam.waterrho, 0.333333);
}
__device__ inline float getMassfromR(float r)
{
	return dparam.waterrho*M_PI*4.0 / 3 * r*r*r;
}

//计算散度
__global__ void cptdivergence(farray outdiv, farray ux, farray uy, farray uz, charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx <dparam.gnum)
	{
		float div = 0, h = dparam.cellsize.x;
		int i, j, k;
		getijk(i, j, k, idx);

		if (mark[idx] == TYPEFLUID)
			div = (ux(i + 1, j, k) - ux(i, j, k) + uy(i, j + 1, k) - uy(i, j, k) + uz(i, j, k + 1) - uz(i, j, k)) / h;

		outdiv[idx] = div;
	}
}

__device__ inline int clampidx(int i, int j, int k)
{
	i = max(0, min(i, NX - 1));
	j = max(0, min(j, NY - 1));
	k = max(0, min(k, NZ - 1));
	return (i*NZ*NY + j*NZ + k);
}


__device__ inline float trilinear(farray u, float x, float y, float z, int w, int h, int d)
{
	x = fmaxf(0.0f, fminf(x, w));
	y = fmaxf(0.0f, fminf(y, h));
	z = fmaxf(0.0f, fminf(z, d));
	int i = fminf(x, w - 2);
	int j = fminf(y, h - 2);
	int k = fminf(z, d - 2);

	return (k + 1 - z)*((j + 1 - y)*((i + 1 - x)*u(i, j, k) + (x - i)*u(i + 1, j, k)) + (y - j)*((i + 1 - x)*u(i, j + 1, k) + (x - i)*u(i + 1, j + 1, k))) +
		(z - k)*((j + 1 - y)*((i + 1 - x)*u(i, j, k + 1) + (x - i)*u(i + 1, j, k + 1)) + (y - j)*((i + 1 - x)*u(i, j + 1, k + 1) + (x - i)*u(i + 1, j + 1, k + 1)));
}

__device__ float3 getVectorFromGrid(float3 pos, farray phigrax, farray phigray, farray phigraz)
{
	float3 res;
	float x = pos.x, y = pos.y, z = pos.z;
	x /= dparam.cellsize.x;
	y /= dparam.cellsize.y;
	z /= dparam.cellsize.z;

	//注意：ux,uy,uz的存储方式比较特殊(staggered grid)，三维线性插值也要比较小心
	res.x = trilinear(phigrax, x - 0.5f, y - 0.5f, z - 0.5f, NX, NY, NZ);
	res.y = trilinear(phigray, x - 0.5f, y - 0.5f, z - 0.5f, NX, NY, NZ);
	res.z = trilinear(phigraz, x - 0.5f, y - 0.5f, z - 0.5f, NX, NY, NZ);
	return res;
}

__device__ float getScaleFromFrid(float3 pos, farray phi)
{
	float res;
	float x = pos.x, y = pos.y, z = pos.z;
	x /= dparam.cellsize.x;
	y /= dparam.cellsize.y;
	z /= dparam.cellsize.z;

	//注意：ux,uy,uz的存储方式比较特殊(staggered grid)，三维线性插值也要比较小心
	res = trilinear(phi, x - 0.5f, y - 0.5f, z - 0.5f, NX, NY, NZ);

	return res;
}

//Jacobi iteration: Ax=b
//todo: check this function and maybe get another solver.
__global__ void JacobiIter(farray outp, farray p, farray b, charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx <dparam.gnum)
	{
		float resp = 0, h = dparam.cellsize.x;
		float p1, p2, p3, p4, p5, p6;
		float p0 = p[idx];
		int i, j, k;

		if (mark[idx] == TYPEFLUID)
		{
			getijk(i, j, k, idx);
			p1 = (mark(i + 1, j, k) == TYPEBOUNDARY) ? p0 : p(i + 1, j, k);
			p2 = (mark(i, j + 1, k) == TYPEBOUNDARY) ? p0 : p(i, j + 1, k);
			p3 = (mark(i, j, k + 1) == TYPEBOUNDARY) ? p0 : p(i, j, k + 1);
			p4 = (mark(i - 1, j, k) == TYPEBOUNDARY) ? p0 : p(i - 1, j, k);
			p5 = (mark(i, j - 1, k) == TYPEBOUNDARY) ? p0 : p(i, j - 1, k);
			p6 = (mark(i, j, k - 1) == TYPEBOUNDARY) ? p0 : p(i, j, k - 1);

			resp = (p1 + p2 + p3 + p4 + p5 + p6 - h*h*b(i, j, k)) / 6.0f;
		}
		outp[idx] = resp;
	}
}

__global__ void setPressBoundary(farray press)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx <dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		if (i == 0) press[idx] = press(i + 1, j, k);
		if (j == 0) press[idx] = press(i, j + 1, k);
		if (k == 0) press[idx] = press(i, j, k + 1);
		if (i == NX - 1) press[idx] = press(i - 1, j, k);
		if (j == NY - 1) press[idx] = press(i, j - 1, k);
		if (k == NZ - 1) press[idx] = press(i, j, k - 1);
	}
}

//压强与速度的计算
__global__ void subGradPress(farray p, farray ux, farray uy, farray uz)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float h = dparam.cellsize.x;
	if (idx<dparam.gvnum.x)
	{
		//ux
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		if (i>0 && i<NX)		//look out for this condition
			ux(i, j, k) -= (p(i, j, k) - p(i - 1, j, k)) / h;
	}
	if (idx<dparam.gvnum.y)
	{
		//uy
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		if (j>0 && j<NY)		//look out for this condition
			uy(i, j, k) -= (p(i, j, k) - p(i, j - 1, k)) / h;
	}
	if (idx<dparam.gvnum.z)
	{
		//uz
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		if (k>0 && k<NZ)		//look out for this condition
			uz(i, j, k) -= (p(i, j, k) - p(i, j, k - 1)) / h;
	}
}


__device__ float3 getParticleVelFromGrid(float3 pos, farray ux, farray uy, farray uz)
{
	float3 vel;
	float x = pos.x, y = pos.y, z = pos.z;
	x /= dparam.cellsize.x;
	y /= dparam.cellsize.y;
	z /= dparam.cellsize.z;

	//注意：ux,uy,uz的存储方式比较特殊(staggered grid)，三维线性插值也要比较小心
	vel.x = trilinear(ux, x, y - 0.5f, z - 0.5f, NX + 1, NY, NZ);
	vel.y = trilinear(uy, x - 0.5f, y, z - 0.5f, NX, NY + 1, NZ);
	vel.z = trilinear(uz, x - 0.5f, y - 0.5f, z, NX, NY, NZ + 1);
	return vel;
}

__global__ void mapvelg2p_flip(float3 *ppos, float3 *vel, char* parflag, int pnum, farray ux, farray uy, farray uz)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//pos-->grid xyz
		float3 ipos = ppos[idx];
		float3 gvel = getParticleVelFromGrid(ipos, ux, uy, uz);

		vel[idx] += gvel;
	}
}

__device__ inline float sharp_kernel(float r2, float h)
{
	return fmax(h*h / fmax(r2, 0.0001f) - 1.0f, 0.0f);
}

__global__ void mapvelp2g_slow(float3 *pos, float3 *vel, int pnum, farray ux, farray uy, farray uz)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float w, weight, RE = 1.4, dis2, usum;
	float3 gpos;
	float scale = 1 / dparam.cellsize.x;
	if (idx<dparam.gvnum.x)
	{
		// ux
		weight = 0, usum = 0;
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		gpos.x = i, gpos.y = j + 0.5, gpos.z = k + 0.5;
		for (int p = 0; p<pnum; p++)
		{
			dis2 = dot(pos[p] * scale - gpos, pos[p] * scale - gpos);
			w = sharp_kernel(dis2, RE);
			weight += w;
			usum += w*vel[p].x;
		}
		usum = (weight>0) ? (usum / weight) : 0.0f;
		ux(i, j, k) = usum;
	}
	if (idx<dparam.gvnum.y)
	{
		// uy
		weight = 0, usum = 0;
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		gpos.x = i + 0.5, gpos.y = j, gpos.z = k + 0.5;
		for (int p = 0; p<pnum; p++)
		{
			dis2 = dot((pos[p] * scale) - gpos, (pos[p] * scale) - gpos);
			w = sharp_kernel(dis2, RE);
			weight += w;
			usum += w*vel[p].y;
		}
		usum = (weight>0) ? (usum / weight) : 0.0f;
		uy(i, j, k) = usum;
	}
	if (idx<dparam.gvnum.z)
	{
		// uz
		weight = 0, usum = 0;
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		gpos.x = i + 0.5, gpos.y = j + 0.5, gpos.z = k;
		for (int p = 0; p<pnum; p++)
		{
			dis2 = dot(pos[p] * scale - gpos, pos[p] * scale - gpos);
			w = sharp_kernel(dis2, RE);
			weight += w;
			usum += w*vel[p].z;
		}
		usum = (weight>0.00001) ? (usum / weight) : 0.0f;
		uz(i, j, k) = usum;
	}
}

__device__ inline bool verifycellidx(int i, int j, int k)
{
	if (i<0 || i>NX - 1 || j<0 || j>NY - 1 || k<0 || k>NZ - 1)
		return false;
	return true;
}
__device__ inline bool verifycellidx(int i, int j, int k, int w, int h, int d)
{
	if (i<0 || i>w - 1 || j<0 || j>h - 1 || k<0 || k>d - 1)
		return false;
	return true;
}

__global__ void addgravityforce_k(float3 *vel, char* parflag, int pnum, float dt)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (parflag[idx] == TYPEFLUID || parflag[idx] == TYPESOLID)
			vel[idx] += dt*dparam.gravity;
	}
}

__global__ void addbuoyancyforce_k(float dheight, float3 *pos, float3 *vel, char* parflag, int pnum, float dt)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (parflag[idx] == TYPEAIR)
			vel[idx] -= dt*dparam.gravity * 1.1f;		//todo:这里的浮力可以小一些，让气泡上的慢一些，视频快一些，水看起来就不太粘了。
		else if (parflag[idx] == TYPEAIRSOLO)
			vel[idx] -= dt*dparam.gravity * 1.1f;
		else if (parflag[idx] == TYPESOLID)
			vel[idx] -= dt*dparam.gravity * 0.55f;
		// 		else if(parflag[idx] == TYPESOLID && pos[idx].z <= dheight)			//	液面下固体粒子受浮力
		// 			vel[idx] -= dt*dparam.gravity * 0.2f;
	}
}

__global__ void addbuoyancyforce_vel(float velMax, float3 *pos, float3 *vel, char* parflag, int pnum, float dt, float buoyanceRateAir, float buoyanceRateSolo)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		float rate = fmax(velMax - vel[idx].z, 0.0f) / velMax;
		if (parflag[idx] == TYPEAIR)
			vel[idx].z -= dt*dparam.gravity.z * rate * buoyanceRateAir;		//todo:这里的浮力可以小一些，让气泡上的慢一些，视频快一些，水看起来就不太粘了。
		else if (parflag[idx] == TYPEAIRSOLO)
			vel[idx].z -= dt*dparam.gravity.z *rate* buoyanceRateSolo;
		else if (parflag[idx] == TYPESOLID)
		vel[idx].z += dt*dparam.gravity.z * 0.1f;//0.55f;
		// 		else if(parflag[idx] == TYPESOLID && pos[idx].z <= dheight)			//	液面下固体粒子受浮力
		// 			vel[idx] -= dt*dparam.gravity * 0.2f;
	}
}

__global__ void advectparticle(float3 *ppos, float3 *pvel, int pnum, farray ux, farray uy, farray uz, float dt,
	char *parflag, VELOCITYMODEL velmode)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//read in
		float3 ipos = ppos[idx], ivel = pvel[idx];
		float3 tmin = dparam.gmin + (dparam.cellsize + make_float3(0.5f*dparam.samplespace));
		float3 tmax = dparam.gmax - (dparam.cellsize + make_float3(0.5f*dparam.samplespace));

		//pos-->grid xyz
		float3 gvel;
		gvel = getParticleVelFromGrid(ipos, ux, uy, uz);

		//vel[idx] += dt*dparam.gravity;
		ipos += gvel*dt;

		if (velmode == CIP)
			ivel = gvel;
		else if (velmode == FLIP)
			ivel = (1 - FLIP_ALPHA)*gvel + FLIP_ALPHA*pvel[idx];

		//check boundary
		ipos.x = fmax(tmin.x, fmin(tmax.x, ipos.x));
		ipos.y = fmax(tmin.y, fmin(tmax.y, ipos.y));
		ipos.z = fmax(tmin.z, ipos.z);
		if (ipos.z >= tmax.z)
			ipos.z = tmax.z, ivel.z = 0.0f;

		//write back
		pvel[idx] = ivel;
		ppos[idx] = ipos;
	}
}

__global__ void advectparticle_RK2(float3 *ppos, float3 *pvel, int pnum, farray ux, farray uy, farray uz, float dt,
	char *parflag, VELOCITYMODEL velmode)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//read in
		float3 ipos = ppos[idx], ivel = pvel[idx];
		float3 tmin = dparam.gmin + (dparam.cellsize + make_float3(0.5f*dparam.samplespace));
		float3 tmax = dparam.gmax - (dparam.cellsize + make_float3(0.5f*dparam.samplespace));

		//pos-->grid xyz
		float3 gvel;
		gvel = getParticleVelFromGrid(ipos, ux, uy, uz);

		if (velmode == CIP)
			ivel = gvel;
		else if (velmode == FLIP)
			ivel = (1 - FLIP_ALPHA)*gvel + FLIP_ALPHA*pvel[idx];

		//mid point: x(n+1/2) = x(n) + 0.5*dt*u(xn)
		float3 midpoint = ipos + gvel * dt * 0.5;
		float3 gvelmidpoint = getParticleVelFromGrid(midpoint, ux, uy, uz);
		// x(n+1) = x(n) + dt*u(x+1/2)
		ipos += gvelmidpoint * dt;

		//check boundary
		if (ipos.x <= tmin.x)
			ipos.x = tmin.x, ivel.x = 0.0f;
		if (ipos.y <= tmin.y)
			ipos.y = tmin.y, ivel.y = 0.0f;
		if (ipos.z <= tmin.z)
			ipos.z = tmin.z, ivel.z = 0.0f;

		if (ipos.x >= tmax.x)
			ipos.x = tmax.x, ivel.x = 0.0f;
		if (ipos.y >= tmax.y)
			ipos.y = tmax.y, ivel.y = 0.0f;
		if (ipos.z >= tmax.z)
			ipos.z = tmax.z, ivel.z = 0.0f;

		//write back
		if (parflag[idx] != TYPESOLID)
		{
			pvel[idx] = ivel;
			ppos[idx] = ipos;
		}
		else
			pvel[idx] = ivel;
	}
}

__global__ void flipAirVacuum(charray mark)
{
	uint idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		if (mark[idx] == TYPEVACUUM)
			mark[idx] = TYPEAIR;
	}
}
__global__ void markair(charray mark)
{
	uint idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		mark[idx] = TYPEAIR;
	}
}


__global__ void markforsmoke(charray mark, farray spraydense)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		/*		if(spraydense[idx]>0 )*/
		mark[idx] = TYPEFLUID;
	}
}

__global__ void markfluid(charray mark, float3 *pos, char *parflag, int pnum)
{
	uint idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		int i, j, k;
		//todo: ???? Should spray particle count??? or should we have a more accurate mark method.
		//	if( parflag[idx]==TYPEFLUID)
		{
			getijkfrompos(i, j, k, pos[idx]);
			mark(i, j, k) = TYPEFLUID;		//应该是不需要原子操作的，重复写不会有问题
		}
	}
}

//判断一下格子里含有的fluid particle的数量，再决定格子的属性
__global__ void markfluid_dense(charray mark, float *parmass, char *parflag, int pnum, uint *gridstart, uint *gridend, int fluidParCntPerGridThres)
{
	uint idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int cntfluidsolid = 0, cntair = 0;

		uint start = gridstart[idx];
		uint end = gridend[idx];
		if (start != CELL_UNDEF)
		{
			for (uint p = start; p<end; ++p)
			{
				if (parflag[p] == TYPEFLUID || parflag[p] == TYPESOLID)
					cntfluidsolid++;
				else if (parflag[p] == TYPEAIR)
					cntair++;
			}
		}

		if (cntfluidsolid == 0 && cntair == 0)
			mark[idx] = TYPEVACUUM;
		else if (cntfluidsolid>cntair)
			mark[idx] = TYPEFLUID;
		else
			mark[idx] = TYPEAIR;
	}
}

__global__ void markBoundaryCell(charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1)
			mark[idx] = TYPEBOUNDARY;
	}
}

__global__ void setgridcolor_k(float* color, ECOLORMODE mode,
	farray p, farray ux, farray uy, farray uz, farray div,
	farray phi, charray mark, farray ls, farray tp, float sigma, float temperatureMax, float temperatureMin)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		float3 rescolor = make_float3(0.0);
		int cellindex = NY / 2;
		if (mode == COLOR_PRESS)
		{
			if (j != cellindex || p[idx] == 0)
				rescolor = make_float3(0, 0, 1);
			else if (p[idx]>0)
				rescolor = make_float3(0, 1, 0);
			else if (p[idx]<0)
				rescolor = make_float3(1, 0, 0);
			//rescolor = mapColorBlue2Red( 30000*abs(p[idx]) );
		}

		else if (mode == COLOR_UX)
		{
			if (j != cellindex || ux(i + 1, j, k) + ux(i, j, k)<0)
				rescolor = make_float3(0, 0, 1);
			else
				rescolor = mapColorBlue2Red(0.5*abs(ux(i + 1, j, k) + ux(i, j, k)));
		}
		else if (mode == COLOR_UY)
		{
			if (j != cellindex || uy(i, j + 1, k) + uy(i, j, k)<0)
				rescolor = make_float3(0, 0, 1);
			else
				rescolor = mapColorBlue2Red(0.5*abs(uy(i, j + 1, k) + uy(i, j, k)));
		}
		else if (mode == COLOR_UZ)
		{
			if (j != cellindex/*||uz(i,j,k+1)+uz(i,j,k)<0*/)
				rescolor = make_float3(0, 0, 1);
			else
				rescolor = mapColorBlue2Red(5 * abs(uz(i, j, k)));
		}
		else if (mode == COLOR_DIV)
		{
			if (j != cellindex || div[idx] == 0)
				rescolor = make_float3(0, 0, 1);
			else if (div[idx]>0)
				rescolor = make_float3(0, 1, 0);
			else if (div[idx]<0)
				rescolor = make_float3(1, 1, 0);
		}
		else if (mode == COLOR_PHI)
		{
			if (phi[idx]>3 * NX - 1 || j != cellindex)
				rescolor = make_float3(0, 0, 1);
			else
				rescolor = mapColorBlue2Red(0.5f + phi[idx]);
		}
		else if (mode == COLOR_MARK)
		{
			if (j != cellindex)
				rescolor = make_float3(0, 0, 1);
			else
			{
				if (mark[idx] == TYPEAIR)
					rescolor = make_float3(0, 1, 0);
				else if (mark[idx] == TYPEFLUID)
					rescolor = make_float3(1, 0, 0);
				else if (mark[idx] == TYPEVACUUM)
					rescolor = make_float3(1, 1, 0);
				else if (mark[idx] == TYPEBOUNDARY)
					rescolor = make_float3(0, 1, 1);
				else
					rescolor = make_float3(0, 0, 1);
				//rescolor = mapColorBlue2Red( (int)(mark[idx])+1.0f ) ;
			}
		}
		else if (mode == COLOR_LS)
		{
			if (j == cellindex && ls[idx]>0)
				rescolor = mapColorBlue2Red(abs(ls[idx] / dparam.cellsize.x));
			else
				rescolor = make_float3(0, 0, 1);
		}
		else if (mode == COLOR_TP)
		{
			if (j != cellindex || i == 0 || i == NX - 1 || k == 0 || k == NZ - 1)
				rescolor = make_float3(0, 0, 1);
			else
				//	rescolor = mapColorBlue2Red( abs(tp[idx]*dparam.cellsize.x*5/sigma) );

				//rescolor = mapColorBlue2Red( abs(tp[idx]-353)/5.0f );
				rescolor = mapColorBlue2Red((tp[idx] - temperatureMin) / (temperatureMax - temperatureMin)*6.0f);
		}
		color[idx * 3] = rescolor.x;
		color[idx * 3 + 1] = rescolor.y;
		color[idx * 3 + 2] = rescolor.z;
	}
}

__host__ __device__ inline float3 mapColorBlue2Red(float v)
{
	float3 color;
	if (v<0)
		return make_float3(0.0f, 0.0f, 1.0f);

	int ic = (int)v;
	float f = v - ic;
	switch (ic)
	{
	case 0:
	{
			  color.x = 0;
			  color.y = f / 2;
			  color.z = 1;
	}
		break;
	case 1:
	{

			  color.x = 0;
			  color.y = f / 2 + 0.5f;
			  color.z = 1;
	}
		break;
	case 2:
	{
			  color.x = f / 2;
			  color.y = 1;
			  color.z = 1 - f / 2;
	}
		break;
	case 3:
	{
			  color.x = f / 2 + 0.5f;
			  color.y = 1;
			  color.z = 0.5f - f / 2;
	}
		break;
	case 4:
	{
			  color.x = 1;
			  color.y = 1.0f - f / 2;
			  color.z = 0;
	}
		break;
	case 5:
	{
			  color.x = 1;
			  color.y = 0.5f - f / 2;
			  color.z = 0;
	}
		break;
	default:
	{
			   color.x = 1;
			   color.y = 0;
			   color.z = 0;
	}
		break;
	}
	return color;
}

__global__ void initphi(farray phi, charray mark, char typeflag)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		if (mark[idx] == typeflag)
			phi[idx] = -0.5;
		else
			phi[idx] = NX * 3;
	}
}

__global__ void initSolidPhi(farray phi, uint *gridstart, uint *gridend, char *pflag)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		bool flag = false;
		uint start = gridstart[idx];
		if (start != CELL_UNDEF)
		{
			for (; start<gridend[idx]; start++)
			{
				if (pflag[start] == TYPESOLID)
					flag = true;
			}
		}
		if (flag)
			phi[idx] = -0.5f;
		else
			phi[idx] = 3 * NX;
	}
}

__device__ void solvedistance(float a, float b, float c, float &x)
{
	float d = fmin(a, fmin(b, c)) + 1;
	if (d>fmax(a, fmax(b, c)))
	{
		d = (a + b + c + sqrt(3 - (a - b)*(a - b) - (a - c)*(a - c) - (b - c)*(b - c))) / 3;
	}
	if (d<x) x = d;
}
__global__ void sweepphi(farray phi)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		float resphi = phi[idx];
		for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
		{
			if (verifycellidx(i + di, j, k) && verifycellidx(i, j + dj, k) && verifycellidx(i, j, k + dk))
				solvedistance(phi(i + di, j, k), phi(i, j + dj, k), phi(i, j, k + dk), resphi);
		}
		phi[idx] = resphi;
	}
}
__global__ void sweepphibytype(farray phi, charray mark, char typeflag)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		if (mark[idx] == typeflag)
			return;
		int i, j, k;
		getijk(i, j, k, idx);
		float resphi = phi[idx];
		for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
		{
			if (verifycellidx(i + di, j, k) && verifycellidx(i, j + dj, k) && verifycellidx(i, j, k + dk))
				solvedistance(phi(i + di, j, k), phi(i, j + dj, k), phi(i, j, k + dk), resphi);
		}
		phi[idx] = resphi;
	}
}

__global__ void sweepu(farray outux, farray outuy, farray outuz, farray ux, farray uy, farray uz, farray phi, charray mark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	int i, j, k;
	float wx, wy, wz, wsum;		//三个方向上的权重
	if (idx < dparam.gvnum.x)
	{
		//copy
		outux[idx] = ux[idx];

		//ux
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		if (i>1 && i<NX - 1 /*&& j>0 && j<N-1 && k>0 && k<N-1*/)
		{
			if ((mark(i, j, k) == TYPEAIR && mark(i - 1, j, k) == TYPEAIR) || (mark(i, j, k) == TYPEBOUNDARY && mark(i - 1, j, k) == TYPEBOUNDARY))
			for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
			{
				if (j + dj<0 || j + dj>NY - 1 || k + dk<0 || k + dk >NZ - 1)
					continue;
				wx = -di*(phi(i, j, k) - phi(i - 1, j, k));
				if (wx<0)
					continue;
				wy = (phi(i, j, k) + phi(i - 1, j, k) - phi(i, j + dj, k) - phi(i - 1, j + dj, k))*0.5f;
				if (wy<0)
					continue;
				wz = (phi(i, j, k) + phi(i - 1, j, k) - phi(i, j, k + dk) - phi(i - 1, j, k + dk))*0.5f;
				if (wz<0)
					continue;
				wsum = wx + wy + wz;
				if (wsum == 0)
					wx = wy = wz = 1.0f / 3;
				else
					wx /= wsum, wy /= wsum, wz /= wsum;
				outux(i, j, k) = wx*ux(i + di, j, k) + wy* ux(i, j + dj, k) + wz* ux(i, j, k + dk);
			}
		}
	}
	if (idx < dparam.gvnum.y)
	{
		//copy
		outuy[idx] = uy[idx];

		//uy
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		if ( /*i>0 && i<N-1 &&*/ j>1 && j<NY - 1 /*&& k>0 && k<N-1*/)
		{
			if ((mark(i, j, k) == TYPEAIR && mark(i, j - 1, k) == TYPEAIR) || (mark(i, j, k) == TYPEBOUNDARY && mark(i, j - 1, k) == TYPEBOUNDARY))
			for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
			{
				if (i + di<0 || i + di>NX - 1 || k + dk<0 || k + dk >NZ - 1)
					continue;
				wy = -dj*(phi(i, j, k) - phi(i, j - 1, k));
				if (wy<0)
					continue;
				wx = (phi(i, j, k) + phi(i, j - 1, k) - phi(i + di, j, k) - phi(i + di, j - 1, k))*0.5f;
				if (wx<0)
					continue;
				wz = (phi(i, j, k) + phi(i, j - 1, k) - phi(i, j, k + dk) - phi(i, j - 1, k + dk))*0.5f;
				if (wz<0)
					continue;
				wsum = wx + wy + wz;
				if (wsum == 0)
					wx = wy = wz = 1.0f / 3;
				else
					wx /= wsum, wy /= wsum, wz /= wsum;
				outuy(i, j, k) = wx*uy(i + di, j, k) + wy* uy(i, j + dj, k) + wz* uy(i, j, k + dk);
			}
		}
	}
	if (idx < dparam.gvnum.z)
	{
		//copy
		outuz[idx] = uz[idx];

		//uz
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		if ( /*i>0 && i<N-1 && j>0 && j<N-1 &&*/ k>1 && k<NZ - 1)
		{
			if ((mark(i, j, k) == TYPEAIR && mark(i, j, k - 1) == TYPEAIR) || (mark(i, j, k) == TYPEBOUNDARY && mark(i, j, k - 1) == TYPEBOUNDARY))
			for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
			{
				if (i + di<0 || i + di >NX - 1 || j + dj<0 || j + dj>NY - 1)
					continue;
				wz = -dk*(phi(i, j, k) - phi(i, j, k - 1));
				if (wz<0)
					continue;
				wy = (phi(i, j, k) + phi(i, j, k - 1) - phi(i, j + dj, k) - phi(i, j + dj, k - 1))*0.5f;
				if (wy<0)
					continue;
				wx = (phi(i, j, k) + phi(i, j, k - 1) - phi(i + di, j, k) - phi(i + di, j, k - 1))*0.5f;
				if (wx<0)
					continue;
				wsum = wx + wy + wz;
				if (wsum == 0)
					wx = wy = wz = 1.0f / 3;
				else
					wx /= wsum, wy /= wsum, wz /= wsum;
				outuz(i, j, k) = wx*uz(i + di, j, k) + wy* uz(i, j + dj, k) + wz* uz(i, j, k + dk);
			}
		}
	}
}

__global__ void setSmokeBoundaryU_k(farray ux, farray uy, farray uz, charray mark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	int i, j, k;
	if (idx < dparam.gvnum.x)
	{
		//ux
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		{
			if (i <= 1 || i >= ux.xn - 2)
				ux(i, j, k) = 0.0f;
			else if (j == 0)
				ux(i, j, k) = ux(i, j + 1, k);
			else if (j == NY - 1)
				ux(i, j, k) = ux(i, j - 1, k);
			else if (k == 0)
				ux(i, j, k) = ux(i, j, k + 1);
			else if (k == NZ - 1)
				ux(i, j, k) = ux(i, j, k - 1);
			else if (i>1 && i<NX - 1 && ((mark(i, j, k) == TYPEBOUNDARY) != (mark(i - 1, j, k) == TYPEBOUNDARY)))
				ux(i, j, k) = 0.0f;
		}
	}
	if (idx < dparam.gvnum.y)
	{
		//uy
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		{
			if (j <= 1 || j >= uy.yn - 2)
				uy(i, j, k) = 0.0f;
			else if (i == 0)
				uy(i, j, k) = uy(i + 1, j, k);
			else if (i == NX - 1)
				uy(i, j, k) = uy(i - 1, j, k);
			else if (k == 0)
				uy(i, j, k) = uy(i, j, k + 1);
			else if (k == NZ - 1)
				uy(i, j, k) = uy(i, j, k - 1);
			else if (j>0 && j<NY && ((mark(i, j, k) == TYPEBOUNDARY) != (mark(i, j - 1, k) == TYPEBOUNDARY)))
				uy(i, j, k) = 0.0f;
		}
	}
	if (idx < dparam.gvnum.z)
	{
		//uz
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		{
			if (k <= 1 || k >= uz.zn - 2)
				uz(i, j, k) = 0.0f;
			else if (i == 0)
				uz(i, j, k) = uz(i + 1, j, k);
			else if (i == NX - 1)
				uz(i, j, k) = uz(i - 1, j, k);
			else if (j == 0)
				uz(i, j, k) = uz(i, j + 1, k);
			else if (j == NY - 1)
				uz(i, j, k) = uz(i, j - 1, k);
			else if (k>0 && k<NZ && ((mark(i, j, k) == TYPEBOUNDARY) != (mark(i, j, k - 1) == TYPEBOUNDARY)))
				uz(i, j, k) = 0.0f;
		}
	}
}

__global__ void setWaterBoundaryU_k(farray ux, farray uy, farray uz, charray mark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	int i, j, k;
	if (idx < dparam.gvnum.x)
	{
		//ux
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		{
			if (i <= 1 || i >= ux.xn - 2)
				ux(i, j, k) = 0.0f;
			else if (i>1 && i<NX - 1 && ((mark(i, j, k) == TYPEBOUNDARY) != (mark(i - 1, j, k) == TYPEBOUNDARY)))
				ux(i, j, k) = 0.0f;
		}
	}
	if (idx < dparam.gvnum.y)
	{
		//uy
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		{
			if (j <= 1 || j >= uy.yn - 2)
				uy(i, j, k) = 0.0f;
			else if (j>0 && j<NY && ((mark(i, j, k) == TYPEBOUNDARY) != (mark(i, j - 1, k) == TYPEBOUNDARY)))
				uy(i, j, k) = 0.0f;
		}
	}
	if (idx < dparam.gvnum.z)
	{
		//uz
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		{
			if (k <= 1 || k >= uz.zn - 1)		//特殊处理ceiling
				uz(i, j, k) = 0.0f;
			else if (k == uz.zn - 2)	//ceiling.
				uz(i, j, k) = (uz(i, j, k - 1)<0) ? (uz(i, j, k - 1)) : 0;
			else if (k>0 && k<NZ && ((mark(i, j, k) == TYPEBOUNDARY) != (mark(i, j, k - 1) == TYPEBOUNDARY)))
				uz(i, j, k) = 0.0f;
		}
	}
}

__global__ void computeDeltaU(farray ux, farray uy, farray uz, farray uxold, farray uyold, farray uzold)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gvnum.x)
		uxold[idx] = ux[idx] - uxold[idx];
	if (idx < dparam.gvnum.y)
		uyold[idx] = uy[idx] - uyold[idx];
	if (idx < dparam.gvnum.z)
		uzold[idx] = uz[idx] - uzold[idx];
}


// From CUDA SDK: calculate grid hash value for each particle
__global__ void calcHashD(uint*   gridParticleHash,  // output
	uint*   gridParticleIndex, // output
	float3* pos,               // input: positions
	uint    numParticles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numParticles) return;

	float3 p = pos[index];

	// get address in grid
	int i, j, k;
	getijkfrompos(i, j, k, p);
	int gridindex = getidx(i, j, k);

	// store grid hash and particle index
	gridParticleHash[index] = gridindex;
	gridParticleIndex[index] = index;
}
// From CUDA SDK: calculate grid hash value for each particle
__global__ void calcHashD_MC(uint*   gridParticleHash,  // output
	uint*   gridParticleIndex, // output
	float3* pos,               // input: positions
	uint    numParticles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numParticles) return;
	
	float3 p = pos[index];

	// get address in grid
	int i, j, k;
	getijkfrompos(i, j, k, p, NXMC, NYMC, NZMC, dparam.cellsize.x / NXMC*NX);
	int gridindex = getidx(i, j, k, NXMC, NYMC, NZMC);

	// store grid hash and particle index
	gridParticleHash[index] = gridindex;
	gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__ void reorderDataAndFindCellStartD(uint*   cellStart,        // output: cell start index
	uint*   cellEnd,          // output: cell end index
	float3* sortedPos,        // output: sorted positions
	float3* sortedVel,        // output: sorted velocities
	char* sortedflag,
	float* sortedmass,
	float* sortedTemperature,
	float* sortedheat,
	float* sortedsolubility,
	float* sortedgascontain,
	uint *  gridParticleHash, // input: sorted grid hashes
	uint *  gridParticleIndex,// input: sorted particle indices
	float3* oldPos,           // input: sorted position array
	float3* oldVel,           // input: sorted velocity array
	char* oldflag,
	float* oldmass,
	float* oldtemperature,
	float* oldheat,
	float* oldsolubility,
	float* oldgascontain,
	uint    numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint hash;
	// handle case when no. of particles not multiple of block size
	if (index < numParticles) {
		hash = gridParticleHash[index];

		// Load hash data into shared memory so that we can look 
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index - 1];
		}
	}

	__syncthreads();

	if (index < numParticles) {
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
		float3 pos = oldPos[sortedIndex];       // macro does either global read or texture fetch
		float3 vel = oldVel[sortedIndex];       // see particles_kernel.cuh

		sortedPos[index] = pos;
		sortedVel[index] = vel;
		sortedflag[index] = oldflag[sortedIndex];
		sortedmass[index] = oldmass[sortedIndex];
		sortedTemperature[index] = oldtemperature[sortedIndex];
		sortedheat[index] = oldheat[sortedIndex];
		sortedsolubility[index] = oldsolubility[sortedIndex];
		sortedgascontain[index] = oldgascontain[sortedIndex];
	}
}

__global__ void advectux(farray outux, farray ux, farray uy, farray uz, float velocitydissipation, float3 wind)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gvnum.x)
	{
		//get pos of ux point
		int i, j, k;
		getijk(i, j, k, idx, ux.xn, ux.yn, ux.zn);
		float3 pos = make_float3(i, j + 0.5, k + 0.5);
		//get rid of boundary
		if (i*j*k == 0 || i == NX || j == NY - 1 || k == NZ - 1)
			outux[idx] = 0;
		else
		{
			//get this point's vel, for tracing back.
			float3 vel;
			vel.x = ux[idx];
			vel.y = (uy(i - 1, j, k) + uy(i - 1, j + 1, k) + uy(i, j, k) + uy(i, j + 1, k))*0.25f;
			vel.z = (uz(i - 1, j, k) + uz(i - 1, j, k + 1) + uz(i, j, k) + uz(i, j, k + 1))*0.25f;
			//wind
			vel += wind;
			//get oldpos
			float3 oldpos = pos - dparam.dt*vel / dparam.cellsize.x;		//notice: scale velocity by N, from 0-1 world to 0-N world.
			//get ux
			float oldu = trilinear(ux, oldpos.x, oldpos.y - 0.5f, oldpos.z - 0.5f, ux.xn, ux.yn, ux.zn);
			outux[idx] = oldu * velocitydissipation;
		}
	}
}

__global__ void advectuy(farray outuy, farray ux, farray uy, farray uz, float velocitydissipation, float3 wind)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gvnum.y)
	{
		//get pos of ux point
		int i, j, k;
		getijk(i, j, k, idx, uy.xn, uy.yn, uy.zn);
		float3 pos = make_float3(i + 0.5, j, k + 0.5);
		//get rid of boundary
		if (i*j*k == 0 || i == NX - 1 || j == NY || k == NZ - 1)
			outuy[idx] = 0;
		else
		{
			//get this point's vel, for tracing back.
			float3 vel;
			vel.x = (ux(i, j - 1, k) + ux(i + 1, j - 1, k) + ux(i, j, k) + ux(i + 1, j, k))*0.25f;
			vel.y = uy[idx];
			vel.z = (uz(i, j - 1, k) + uz(i, j - 1, k + 1) + uz(i, j, k) + uz(i, j, k + 1))*0.25f;
			//wind
			vel += wind;
			//get oldpos
			float3 oldpos = pos - dparam.dt*vel / dparam.cellsize.x;		//notice: scale velocity by N, from 0-1 world to 0-N world.
			//get ux
			float oldu = trilinear(uy, oldpos.x - 0.5f, oldpos.y, oldpos.z - 0.5f, uy.xn, uy.yn, uy.zn);
			outuy[idx] = oldu * velocitydissipation;
		}
	}
}

__global__ void advectuz(farray outuz, farray ux, farray uy, farray uz, float velocitydissipation, float3 wind)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gvnum.z)
	{
		//get pos of ux point
		int i, j, k;
		getijk(i, j, k, idx, uz.xn, uz.yn, uz.zn);
		float3 pos = make_float3(i + 0.5, j + 0.5, k);
		//get rid of boundary
		if (i*j*k == 0 || i == NX - 1 || j == NY - 1 || k == NZ)
			outuz[idx] = 0;
		else
		{
			//get this point's vel, for tracing back.
			float3 vel;
			vel.x = (ux(i, j, k - 1) + ux(i + 1, j, k - 1) + ux(i, j, k) + ux(i + 1, j, k))*0.25f;
			vel.y = (uy(i, j, k - 1) + uy(i, j + 1, k - 1) + uy(i, j, k) + uy(i, j + 1, k))*0.25f;
			vel.z = uz[idx];
			//wind
			vel += wind;
			//get oldpos
			float3 oldpos = pos - dparam.dt*vel / dparam.cellsize.x;		//notice: scale velocity by N, from 0-1 world to 0-N world.
			//get ux
			float oldu = trilinear(uz, oldpos.x - 0.5f, oldpos.y - 0.5f, oldpos.z, uz.xn, uz.yn, uz.zn);
			//float oldu = -dparam.dt*3.8f;
			outuz[idx] = oldu * velocitydissipation;
		}
	}
}

__global__ void advectscaler(farray outscalar, farray scalar, farray ux, farray uy, farray uz, float densedissipation, float3 wind)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		//get pos of ux point
		int i, j, k;
		getijk(i, j, k, idx);
		float3 pos = make_float3(i + 0.5, j + 0.5, k + 0.5);
		//get rid of boundary
		if (i*j*k == 0 || i == NX - 1 || j == NY - 1 || k == NZ - 1)
			outscalar[idx] = 0;
		else
		{
			//get this point's vel, for tracing back.
			float3 vel;
			vel.x = (ux(i, j, k) + ux(i + 1, j, k))*0.5f;
			vel.y = (uy(i, j, k) + uy(i, j + 1, k))*0.5f;
			vel.z = (uz(i, j, k) + uz(i, j, k + 1))*0.5f;

			//enforce wind as an external velocity field.
			vel += wind;
			//get oldpos
			float3 oldpos = pos - dparam.dt*vel / dparam.cellsize.x;		//notice: scale velocity by N, from 0-1 world to 0-N world.
			//get ux
			float olds = trilinear(scalar, oldpos.x - 0.5f, oldpos.y - 0.5f, oldpos.z - 0.5f, NX, NY, NZ);
			outscalar[idx] = olds * densedissipation;
		}
	}
}

__global__ void setsmokedense(farray dense)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gvnum.z)
	{
		int i, j, k;
		getijk(i, j, k, idx, dense.xn, dense.yn, dense.zn);
		if (i>28 && i<36 && j>28 && j<36 && k<6)
			dense[idx] = dparam.m0*6.0f;
	}
}

__global__ void setsmokevel(farray uz, farray dense)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gvnum.z)
	{
		int i, j, k;
		getijk(i, j, k, idx, uz.xn, uz.yn, uz.zn);
		// 		if( i>20 && i<40 && j>20 && j<40 && k<10 )
		// 			uz[idx] = 4.0f;

		// 		if( k>1 && k<NZ-1 )
		// 			if( dense(i,j,k-1)>0 )
		// 				uz[idx] = 4.0f;

		if (k>1 && k<NZ - 1)
		{
			float alpha = 1000.0f;
			uz(i, j, k) += alpha * dense(i, j, k - 1);
		}
	}
}

__global__ void setsmokevel_nozzle(farray ux, farray dense)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gvnum.x)
	{
		int i, j, k;
		getijk(i, j, k, idx, ux.xn, ux.yn, ux.zn);
		// 		if( i>20 && i<40 && j>20 && j<40 && k<10 )
		// 			uz[idx] = 4.0f;

		//float alpha = 10000.0f;
		if (i>1 && i<NX - 1)
		if (dense(i - 1, j, k)>0)
			ux[idx] = 8.0f;
		//uz(i,j,k) += alpha * dense(i,j,k-1);
	}
}

surface<void, cudaSurfaceType3D> surfaceWrite;

__global__ void writedens2surface_k(farray dens)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		// 		float4 idens = make_float4( 0.0f );
		// 		if(i>10&&i<50 &&j>10&&j<50&&k>10&&k<50 )
		// 			idens = make_float4( 1.0f );
		float4 idens = make_float4(dens[idx] * 10000);
		surf3Dwrite(idens, surfaceWrite, i*sizeof(float4), j, k);		//why *sizeof(float4)?
	}
}

void writedens2surface(cudaArray* cudaarray, int blocknum, int threadnum, farray dense)
{
	cudaBindSurfaceToArray(surfaceWrite, cudaarray);

	//kernel
	writedens2surface_k << <blocknum, threadnum >> >(dense);
}

__device__ float smooth_kernel(float r2, float h) {
	return fmax(1.0f - r2 / (h*h), 0.0f);
}

__device__ float3 sumcellspring(float3 ipos, float3 *pos, float* pmass, char* parflag, uint *gridstart, uint  *gridend, int gidx, float idiameter)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return make_float3(0.0f);
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dist, w;
	float3 spring = make_float3(0.0f);
	float r = 0;
	for (uint p = start; p<end; ++p)
	{
		//if( parflag[p]!=TYPESOLID )		//solid粒子也应该对别的粒子产生作用才对
		{
			dist = length(pos[p] - ipos);
			r = idiameter;//+getRfromMass( pmass[p] );
			w = pmass[p] * smooth_kernel(dist*dist, r);
			if (dist>0.1f*idiameter)	//太近会产生非常大的弹力
				spring += w*(ipos - pos[p]) / dist;
		}
	}
	return spring;
}

__global__ void correctparticlepos(float3* outpos, float3* ppos, float *pmass, char* parflag, int pnum,
	uint* gridstart, uint *gridend, float correctionspring, float correctionradius, float3 *pepos, float *peradius, int penum)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (parflag[idx] == TYPESOLID/* || parflag[idx]==TYPEAIR*/ || parflag[idx] == TYPEAIRSOLO)
		{
			outpos[idx] = ppos[idx];
			return;
		}
		float3 ipos = ppos[idx];
		int i, j, k;
		getijkfrompos(i, j, k, ipos);
		float3 spring = make_float3(0.0f);
		float3 tmin = dparam.gmin + (dparam.cellsize + make_float3(0.5f*dparam.samplespace));
		float3 tmax = dparam.gmax - (dparam.cellsize + make_float3(0.5f*dparam.samplespace));

		float re = correctionradius*dparam.cellsize.x;
		//	float re= getRfromMass( pmass[idx] );
		int lv = 1;
		//	float idiameter = 2*pow(0.75*pmass[idx]/dparam.waterrho/M_PI, 1.0/3);		//注意，应该比实际的半径大，相当于SPH中的核函数半径
		for (int di = -lv; di <= lv; di++) for (int dj = -lv; dj <= lv; dj++) for (int dk = -lv; dk <= lv; dk++)
		{
			if (verifycellidx(i + di, j + dj, k + dk))
			{
				spring += sumcellspring(ipos, ppos, pmass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk), re);
			}
		}

		// 		//增加empty气泡的作用，遍历所有的empty粒子
		// 		float w, dist;
		// 		for( int p=0; p<penum; p++ )
		// 		{
		// 			if( peradius[p]>0.5f*dparam.cellsize.x )	//太小不处理
		// 			{
		// 				dist=length(pepos[p]-ipos);
		// 				w = pmass[idx]*smooth_kernel(dist*dist, peradius[p]);	//质量用被弹开粒子的质量
		// 				if( dist>0.1f*peradius[p] )		//太近会产生非常大的弹力
		// 					spring += w*(ipos-pepos[p]) / dist;
		// 			}
		// 		}

		spring *= correctionspring*re;

		if (length(dparam.dt*spring)>0.3f*dparam.cellsize.x)
			ipos += dparam.cellsize.x * 0.3f * spring / length(spring);
		else
			ipos += dparam.dt*spring;
		ipos.x = fmax(tmin.x, fmin(tmax.x, ipos.x));
		ipos.y = fmax(tmin.y, fmin(tmax.y, ipos.y));
		ipos.z = fmax(tmin.z, fmin(tmax.z, ipos.z));
		outpos[idx] = ipos;
	}
}

__device__ void sumcelldens(float &phi, float3 gpos, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, int gidx)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis;
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == TYPEFLUID || parflag[p] == TYPESOLID)
		{
			dis = length(pos[p] - gpos);
			if (phi>dis) phi = dis;
		}
	}
}

//得到网格上每一个结点的密度值，为MC算法做准备
//[2012][TVCG]Preserving Fluid Sheets with Adaptively Sampled Anisotropic Particles
__global__ void genWaterDensfield(farray outdens, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, float fMCDensity)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < (NX + 1)*(NY + 1)*(NZ + 1))
	{
		float h = dparam.cellsize.x;
		float phi = 8 * fMCDensity*h;		//from flip3d_vs

		//get position
		int i, j, k;
		getijk(i, j, k, idx, NX + 1, NY + 1, NZ + 1);

		float3 p = make_float3(i, j, k)*h;
		for (int di = -2; di <= 1; ++di) for (int dj = -2; dj <= 1; ++dj) for (int dk = -2; dk <= 1; ++dk)
		{
			if (verifycellidx(i + di, j + dj, k + dk))
			{
				sumcelldens(phi, p, pos, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
			}
		}
		phi = fMCDensity*h - phi;

		if (i*j*k == 0 || i == NX || j == NY || k == NZ)
			phi = fmin(phi, -0.1f);

		outdens[idx] = phi;
	}
}

__device__ float3 sumcelldens2(float& wsum, float3 gpos, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, int gidx, float R, char MCParType)
{
	float3 res = make_float3(0.0f);
	if (gridstart[gidx] == CELL_UNDEF)
		return res;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis, w;
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == MCParType)
		{
			dis = length(pos[p] - gpos);
			if (dis<R)
			{
				w = R*R - dis*dis;
				w = w*w*w;
				res += pos[p] * w;
				wsum += w;
			}
		}
	}
	return res;
}

//得到网格上每一个结点的密度值，为MC算法做准备
//[2012]【CGF】Parallel Surface Reconstruction for Particle-Based Fluids
__global__ void genWaterDensfield2(farray outdens, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, float fMCDensity, char MCParType)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < (NXMC + 1)*(NYMC + 1)*(NZMC + 1))
	{
		float phi;
		float h = dparam.cellsize.x / (NXMC / NX);
		//todo: this is not quite right, r should be 0.5*samplespace, i.e. 0.25f/gn.
		float r = 1.0f*h;
		//get position
		int i, j, k;
		getijk(i, j, k, idx, NXMC + 1, NYMC + 1, NZMC + 1);

		float3 p = make_float3(i, j, k)* h;	//网格的位置
		float3 center = make_float3(0.0f);
		float wsum = 0.0f;
		int rate = 2;
		for (int di = -2; di <= 1; ++di) for (int dj = -2; dj <= 1; ++dj) for (int dk = -2; dk <= 1; ++dk)
		{
			if (verifycellidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC))
			{
				center += sumcelldens2(wsum, p, pos, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC), h*rate, MCParType);
			}
		}
		if (wsum>0)
		{
			center /= wsum;
			phi = r - length(p - center);
		}
		else
			phi = -r;		//todo: this may change corresponding to grid resolution.

		if (i*j*k == 0 || i == NXMC || j == NYMC || k == NZMC)
			phi = -1000.0f;
		//phi = fmin( phi, -10.0f);

		outdens[idx] = phi;
	}
}

__device__ float3 sumcelldens_Gas(float& wsum, float3 gpos, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, int gidx, float R, SCENE scene)
{
	float3 res = make_float3(0.0f);
	if (gridstart[gidx] == CELL_UNDEF)
		return res;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis, w;
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == TYPEAIR || (parflag[p] == TYPEAIRSOLO && scene != SCENE_INTERACTION))
		{
			dis = length(pos[p] - gpos);
			if (dis<R)
			{
				w = R*R - dis*dis;
				w = w*w*w;
				res += pos[p] * w;
				wsum += w;
			}
		}
	}
	return res;
}

//得到网格上每一个结点的密度值，为MC算法做准备
//[2012]【CGF】Parallel Surface Reconstruction for Particle-Based Fluids
__global__ void genWaterDensfield_Gas(farray outdens, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, float fMCDensity, SCENE scene)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < (NXMC + 1)*(NYMC + 1)*(NZMC + 1))
	{
		float phi;
		float h = dparam.cellsize.x / (NXMC / NX);
		//todo: this is not quite right, r should be 0.5*samplespace, i.e. 0.25f/gn.
		float r = 0.8f*h;
		//get position
		int i, j, k;
		getijk(i, j, k, idx, NXMC + 1, NYMC + 1, NZMC + 1);

		float3 p = make_float3(i, j, k)* h;	//网格的位置
		float3 center = make_float3(0.0f);
		float wsum = 0.0f;
		int rate = 2;
		for (int di = -2; di <= 1; ++di) for (int dj = -2; dj <= 1; ++dj) for (int dk = -2; dk <= 1; ++dk)
		{
			if (verifycellidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC))
			{
				center += sumcelldens_Gas(wsum, p, pos, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC), h*rate, scene);
			}
		}
		if (wsum>0)
		{
			center /= wsum;
			phi = r - length(p - center);
		}
		else
			phi = -r;		//todo: this may change corresponding to grid resolution.

		if (i*j*k == 0 || i == NXMC || j == NYMC || k == NZMC)
			phi = -1000.0f;
		//phi = fmin( phi, -10.0f);

		outdens[idx] = phi;
	}
}

__device__ float3 sumcelldens_liquidAndGas(float& wsum, float3 gpos, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, int gidx, float R)
{
	float3 res = make_float3(0.0f);
	if (gridstart[gidx] == CELL_UNDEF)
		return res;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis, w;
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == TYPEAIR || parflag[p] == TYPEAIRSOLO || parflag[p] == TYPEFLUID)
		{
			dis = length(pos[p] - gpos);
			if (dis<R)
			{
				w = R*R - dis*dis;
				w = w*w*w;
				res += pos[p] * w;
				wsum += w;
			}
		}
	}
	return res;
}

//得到网格上每一个结点的密度值，为MC算法做准备
//[2012]【CGF】Parallel Surface Reconstruction for Particle-Based Fluids
__global__ void genWaterDensfield_liquidAndGas(farray outdens, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, float fMCDensity)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < (NXMC + 1)*(NYMC + 1)*(NZMC + 1))
	{
		float phi;
		float h = dparam.cellsize.x / (NXMC / NX);
		//todo: this is not quite right, r should be 0.5*samplespace, i.e. 0.25f/gn.
		//float r = 1.0f*h;
		float r = 0.25*h;
		//get position
		int i, j, k;
		getijk(i, j, k, idx, NXMC + 1, NYMC + 1, NZMC + 1);

		float3 p = make_float3(i, j, k)* h;	//网格的位置
		float3 center = make_float3(0.0f);
		float wsum = 0.0f;
		int rate = 2;
		for (int di = -2; di <= 1; ++di) for (int dj = -2; dj <= 1; ++dj) for (int dk = -2; dk <= 1; ++dk)
		{
			if (verifycellidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC))
			{
				center += sumcelldens_liquidAndGas(wsum, p, pos, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC), h*rate);
			}
		}
		if (wsum>0)
		{
			center /= wsum;
			phi = r - length(p - center);
		}
		else
			phi = -r;		//todo: this may change corresponding to grid resolution.

		if (i*j*k == 0 || i == NXMC || j == NYMC || k == NZMC)
			phi = -1000.0f;
		//phi = fmin( phi, -10.0f);

		outdens[idx] = phi;
	}
}

__device__ float3  sumcelldens3(float& wsum, float3 gpos, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, int gidx, float h, char MCParType)
{
	float3 res = make_float3(0.0f);
	if (gridstart[gidx] == CELL_UNDEF)
		return res;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis, w;
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == MCParType)
		{
			//GY:参照论文【CFG2012】Parallel Surface Reconstruction for Particle-Based Fluids
			//			  [2007CAVW]A Unified Particle Model for Fluid-Solid Interactions
			//			【2012 VRIPHYS】An Efficient Surface Reconstruction Pipeline for Particle-Based Fluids
			dis = length(pos[p] - gpos);	//v-xi
			if (dis<h)
			{
				// 				w = h*h -dis*dis;  //之前的代码
				// 				w = w*w*w;
				// 				res += pos[p] * w;
				// 				wsum += w;

				w = dis / (4 * h);	//	|v-xi|/R  见[2007 CAVW]下同 R=2h=4r
				w = 1 - w*w;		//	1-s~2
				w = max(w*w*w, 0.0);	//	k(s)
				res += pos[p] * w;
				wsum += w;
			}
		}
	}
	return res;
}

//得到网格上每一个结点的密度值，为MC算法做准备
//[2012]【VRIPHYS】An Efficient Surface Reconstruction Pipeline for Particle-Based Fluids
__global__ void genWaterDensfield_GY(farray outdens, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, float fMCDensity, char MCParType, float3 centertmp)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < (NXMC + 1)*(NYMC + 1)*(NZMC + 1))
	{
		float phi;
		float h = dparam.cellsize.x / (NXMC / NX);
		//todo: this is not quite right, r should be 0.5*samplespace, i.e. 0.25f/gn.
		float r = 0.75f*h;
		float thigh = 0.51;
		float tlow = 0.49;
		//get position
		int i, j, k;
		getijk(i, j, k, idx, NXMC + 1, NYMC + 1, NZMC + 1);

		float3 p = make_float3(i, j, k)* h;	//网格的位置
		float3 center = make_float3(0.0f);
		float wsum = 0.0f;
		for (int di = -2; di <= 1; ++di) for (int dj = -2; dj <= 1; ++dj) for (int dk = -2; dk <= 1; ++dk)
		{
			if (verifycellidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC))
			{
				center += sumcelldens3(wsum, p, pos, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk, NXMC, NYMC, NZMC), h, MCParType);
			}
		}

		if (wsum>0)
		{
			center /= wsum;			//~v

			float3 delta = center - centertmp;
			float Ev = max(delta.x, max(delta.y, delta.z)) / (4 * h);	//
			//	float Ev = 3.8;
			centertmp = center;             //	centertmp:存储的是上一次的center 求Ev的delta用
			float gamma = (thigh - Ev) / (thigh - tlow);
			float f = (Ev<tlow) ? 1 : gamma*gamma*gamma - 3 * gamma*gamma + 3 * gamma;

			//		phi = r - length( p - center );
			phi = (length(p - center) - r*f);
		}
		else
			phi = -r;		//todo: this may change corresponding to grid resolution.

		if (i*j*k == 0 || i == NXMC || j == NYMC || k == NZMC)
			phi = fmin(phi, -10.0f);

		outdens[idx] = phi;
	}
}

__global__ void markSolid_sphere(float3 spherepos, float sphereradius, charray mark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		float3 gpos = (make_float3(i, j, k) + make_float3(0.5f)) * dparam.cellsize.x;
		if (length(gpos - spherepos)<sphereradius)
			mark[idx] = TYPEBOUNDARY;
	}
}

__global__ void markSolid_waterfall(int3 minpos, int3 maxpos, charray mark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		int x, y, z;
		getijk(x, y, z, idx);
		if (x <= maxpos.x && (y >= maxpos.y || y <= minpos.y) && z <= maxpos.z)
			mark[idx] = TYPEBOUNDARY;
		else if (x <= maxpos.x && (y>minpos.y || y<maxpos.y) && z <= minpos.z)
			mark[idx] = TYPEBOUNDARY;
	}
}

//a trick part.
__global__ void markSolid_waterfall_liquid(int3 minpos, int3 maxpos, charray mark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		int x, y, z;
		getijk(x, y, z, idx);
		if (x <= maxpos.x && (y >= maxpos.y || y <= minpos.y) && z <= maxpos.z*0.7f)
			mark[idx] = TYPEBOUNDARY;
		else if (x <= maxpos.x && (y>minpos.y || y<maxpos.y) && z <= minpos.z*0.7f)
			mark[idx] = TYPEBOUNDARY;
	}
}

//a trick part.
__global__ void markSolid_terrain(charray mark, charray mark_terrain)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < dparam.gnum)
	{
		if (mark_terrain[idx] == TYPEBOUNDARY)
			mark[idx] = TYPEBOUNDARY;
	}
}

//得到网格上每一个结点的密度值，为MC算法做准备
__global__ void genSphereDensfield(farray outdens, float3 center, float radius)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < (NXMC + 1)*(NYMC + 1)*(NZMC + 1))
	{
		//float3 center = make_float3(0.5f);
		float phi;

		//get position
		int i, j, k;
		getijk(i, j, k, idx, NXMC + 1, NYMC + 1, NZMC + 1);
		if (i*j*k == 0 || i == NXMC || j == NYMC || k == NZMC)
			phi = -0.1;
		else
		{
			float3 p = make_float3(i, j, k)*dparam.cellsize.x / (NXMC / NX);
			phi = radius - length(p - center);
		}
		outdens[idx] = phi;
	}
}

//-----MC 算法，from cuda sdk 4.2
// classify voxel based on number of vertices it will generate
// one thread per voxel (cell)
__global__ void classifyVoxel(uint* voxelVerts, uint *voxelOccupied, farray volume, float isoValue)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<NXMC*NYMC*NZMC)
	{
		int i, j, k;
		getijk(i, j, k, idx, NXMC, NYMC, NZMC);

		float field[8];
		field[0] = volume(i, j, k);
		field[1] = volume(i + 1, j, k);
		field[2] = volume(i + 1, j + 1, k);
		field[3] = volume(i, j + 1, k);
		field[4] = volume(i, j, k + 1);
		field[5] = volume(i + 1, j, k + 1);
		field[6] = volume(i + 1, j + 1, k + 1);
		field[7] = volume(i, j + 1, k + 1);

		// calculate flag indicating if each vertex is inside or outside isosurface
		uint cubeindex;
		cubeindex = uint(field[0] < isoValue);
		cubeindex += uint(field[1] < isoValue) * 2;
		cubeindex += uint(field[2] < isoValue) * 4;
		cubeindex += uint(field[3] < isoValue) * 8;
		cubeindex += uint(field[4] < isoValue) * 16;
		cubeindex += uint(field[5] < isoValue) * 32;
		cubeindex += uint(field[6] < isoValue) * 64;
		cubeindex += uint(field[7] < isoValue) * 128;

		// read number of vertices from texture
		uint numVerts = tex1Dfetch(numVertsTex, cubeindex);

		voxelVerts[idx] = numVerts;
		voxelOccupied[idx] = (numVerts > 0);
	}//endif
}

// compact voxel array
__global__ void
compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (voxelOccupied[i] && (i < numVoxels)) {
		compactedVoxelArray[voxelOccupiedScan[i]] = i;
	}
}

// compute interpolated vertex along an edge
__device__
float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1)
{
	float t = (isolevel - f0) / (f1 - f0);
	return lerp(p0, p1, t);
}

// calculate triangle normal
__device__
float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
{
	float3 edge0 = *v1 - *v0;
	float3 edge1 = *v2 - *v0;
	// note - it's faster to perform normalization in vertex shader rather than here
	return cross(edge0, edge1);
}


__device__ int GetVertexID(int i, int j, int k)
{
	return 3 * (i*(NZMC + 1)*(NYMC + 1) + j*(NZMC + 1) + k);
}

__device__ int GetEdgeID(int nX, int nY, int nZ, int edge)
{
	//	return GetVertexID( nX,nY,nZ );

	switch (edge) {
	case 0:
		return GetVertexID(nX, nY, nZ) + 1;
	case 1:
		return GetVertexID(nX + 1, nY, nZ);
	case 2:
		return GetVertexID(nX, nY + 1, nZ) + 1;
	case 3:
		return GetVertexID(nX, nY, nZ);
	case 4:
		return GetVertexID(nX, nY, nZ + 1) + 1;
	case 5:
		return GetVertexID(nX + 1, nY, nZ + 1);
	case 6:
		return GetVertexID(nX, nY + 1, nZ + 1) + 1;
	case 7:
		return GetVertexID(nX, nY, nZ + 1);
	case 8:
		return GetVertexID(nX, nY, nZ) + 2;
	case 9:
		return GetVertexID(nX + 1, nY, nZ) + 2;
	case 10:
		return GetVertexID(nX + 1, nY + 1, nZ) + 2;
	case 11:
		return GetVertexID(nX, nY + 1, nZ) + 2;
	default:
		// Invalid edge no.
		return -1;
	}
}

// version that calculates flat surface normal for each triangle
__global__ void
generateTriangles2(float3 *pos, float3 *norm, uint *compactedVoxelArray, uint *numVertsScanned, farray volume,
float isoValue, uint activeVoxels, uint maxVerts)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint idx = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (idx > activeVoxels - 1) {
		idx = activeVoxels - 1;
	}

	int voxel = compactedVoxelArray[idx];
	float3 voxelSize = dparam.cellsize / (NXMC / NX);

	// compute position in 3d grid
	int i, j, k;
	getijk(i, j, k, voxel, NXMC, NYMC, NZMC);

	float3 p;
	p.x = i*voxelSize.x;
	p.y = j*voxelSize.y;
	p.z = k*voxelSize.z;

	float field[8];
	field[0] = volume(i, j, k);
	field[1] = volume(i + 1, j, k);
	field[2] = volume(i + 1, j + 1, k);
	field[3] = volume(i, j + 1, k);
	field[4] = volume(i, j, k + 1);
	field[5] = volume(i + 1, j, k + 1);
	field[6] = volume(i + 1, j + 1, k + 1);
	field[7] = volume(i, j + 1, k + 1);

	// calculate cell vertex positions
	float3 v[8];
	v[0] = p;
	v[1] = p + make_float3(voxelSize.x, 0, 0);
	v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
	v[3] = p + make_float3(0, voxelSize.y, 0);
	v[4] = p + make_float3(0, 0, voxelSize.z);
	v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
	v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
	v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);

	// recalculate flag
	uint cubeindex;
	cubeindex = uint(field[0] < isoValue);
	cubeindex += uint(field[1] < isoValue) * 2;
	cubeindex += uint(field[2] < isoValue) * 4;
	cubeindex += uint(field[3] < isoValue) * 8;
	cubeindex += uint(field[4] < isoValue) * 16;
	cubeindex += uint(field[5] < isoValue) * 32;
	cubeindex += uint(field[6] < isoValue) * 64;
	cubeindex += uint(field[7] < isoValue) * 128;

	// find the vertices where the surface intersects the cube 

	// use shared memory to avoid using local
	__shared__ float3 vertlist[12 * NTHREADS];

	vertlist[threadIdx.x] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
	vertlist[NTHREADS + threadIdx.x] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
	vertlist[(NTHREADS * 2) + threadIdx.x] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
	vertlist[(NTHREADS * 3) + threadIdx.x] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);
	vertlist[(NTHREADS * 4) + threadIdx.x] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
	vertlist[(NTHREADS * 5) + threadIdx.x] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
	vertlist[(NTHREADS * 6) + threadIdx.x] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
	vertlist[(NTHREADS * 7) + threadIdx.x] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);
	vertlist[(NTHREADS * 8) + threadIdx.x] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
	vertlist[(NTHREADS * 9) + threadIdx.x] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
	vertlist[(NTHREADS * 10) + threadIdx.x] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
	vertlist[(NTHREADS * 11) + threadIdx.x] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);
	__syncthreads();

	// output triangle vertices
	uint numVerts = tex1Dfetch(numVertsTex, cubeindex);
	for (int idx2 = 0; idx2<numVerts; idx2 += 3) {
		uint index = numVertsScanned[voxel] + idx2;

		float3 *v[3];
		uint edge;
		edge = tex1Dfetch(triTex, (cubeindex * 16) + idx2);

		v[0] = &vertlist[(edge*NTHREADS) + threadIdx.x];


		edge = tex1Dfetch(triTex, (cubeindex * 16) + idx2 + 1);

		v[1] = &vertlist[(edge*NTHREADS) + threadIdx.x];


		edge = tex1Dfetch(triTex, (cubeindex * 16) + idx2 + 2);

		v[2] = &vertlist[(edge*NTHREADS) + threadIdx.x];

		// calculate triangle surface normal
		float3 n = calcNormal(v[0], v[1], v[2]);

		/*if (index < (maxVerts - 3)) */{
			pos[index] = *v[0];
			norm[index] = n;

			pos[index + 1] = *v[1];
			norm[index + 1] = n;

			pos[index + 2] = *v[2];
			norm[index + 2] = n;
		}
	}
}

// version that calculates flat surface normal for each triangle
__global__ void
generateTriangles_indices(float3 *pTriVertex, uint *pTriIndices, uint *compactedVoxelArray, farray volume,
float isoValue, uint activeVoxels, uint maxVerts, uint *MCEdgeIdxMapped, uint *numVertsScanned)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint idx = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (idx > activeVoxels - 1) {
		idx = activeVoxels - 1;
	}

	int voxel = compactedVoxelArray[idx];
	float3 voxelSize = dparam.cellsize / (NXMC / NX);

	// compute position in 3d grid
	int i, j, k;
	getijk(i, j, k, voxel, NXMC, NYMC, NZMC);

	float3 p;
	p.x = i*voxelSize.x;
	p.y = j*voxelSize.y;
	p.z = k*voxelSize.z;

	float field[8];
	field[0] = volume(i, j, k);
	field[1] = volume(i + 1, j, k);
	field[2] = volume(i + 1, j + 1, k);
	field[3] = volume(i, j + 1, k);
	field[4] = volume(i, j, k + 1);
	field[5] = volume(i + 1, j, k + 1);
	field[6] = volume(i + 1, j + 1, k + 1);
	field[7] = volume(i, j + 1, k + 1);

	// calculate cell vertex positions
	float3 v[8];
	v[0] = p;
	v[1] = p + make_float3(voxelSize.x, 0, 0);
	v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
	v[3] = p + make_float3(0, voxelSize.y, 0);
	v[4] = p + make_float3(0, 0, voxelSize.z);
	v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
	v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
	v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);

	// recalculate flag
	uint cubeindex;
	cubeindex = uint(field[0] < isoValue);
	cubeindex += uint(field[1] < isoValue) * 2;
	cubeindex += uint(field[2] < isoValue) * 4;
	cubeindex += uint(field[3] < isoValue) * 8;
	cubeindex += uint(field[4] < isoValue) * 16;
	cubeindex += uint(field[5] < isoValue) * 32;
	cubeindex += uint(field[6] < isoValue) * 64;
	cubeindex += uint(field[7] < isoValue) * 128;

	// find the vertices where the surface intersects the cube 

	// use shared memory to avoid using local
	__shared__ float3 vertlist[12 * NTHREADS];

	vertlist[threadIdx.x] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
	vertlist[NTHREADS + threadIdx.x] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
	vertlist[(NTHREADS * 2) + threadIdx.x] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
	vertlist[(NTHREADS * 3) + threadIdx.x] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);
	vertlist[(NTHREADS * 4) + threadIdx.x] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
	vertlist[(NTHREADS * 5) + threadIdx.x] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
	vertlist[(NTHREADS * 6) + threadIdx.x] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
	vertlist[(NTHREADS * 7) + threadIdx.x] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);
	vertlist[(NTHREADS * 8) + threadIdx.x] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
	vertlist[(NTHREADS * 9) + threadIdx.x] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
	vertlist[(NTHREADS * 10) + threadIdx.x] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
	vertlist[(NTHREADS * 11) + threadIdx.x] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);
	__syncthreads();

	// output triangle vertices
	uint numVerts = tex1Dfetch(numVertsTex, cubeindex);
	uint edge, mappededgeidx;
	for (int idx2 = 0; idx2<numVerts; idx2 += 3) {
		uint index = numVertsScanned[voxel] + idx2;	//vertex index to write back, sort by each triangle.

		//写入triangle包含的三个顶点的索引，索引是未经过处理的，即边的全局编号，之后单独处理
		edge = tex1Dfetch(triTex, (cubeindex * 16) + idx2);
		mappededgeidx = MCEdgeIdxMapped[GetEdgeID(i, j, k, edge)];
		pTriIndices[index] = mappededgeidx;		//notice: indices begin from 0.
		pTriVertex[mappededgeidx] = (vertlist[(edge*NTHREADS) + threadIdx.x]);

		edge = tex1Dfetch(triTex, (cubeindex * 16) + idx2 + 1);
		mappededgeidx = MCEdgeIdxMapped[GetEdgeID(i, j, k, edge)];
		pTriIndices[index + 1] = mappededgeidx;		//notice: indices begin from 0.
		pTriVertex[mappededgeidx] = (vertlist[(edge*NTHREADS) + threadIdx.x]);

		edge = tex1Dfetch(triTex, (cubeindex * 16) + idx2 + 2);
		mappededgeidx = MCEdgeIdxMapped[GetEdgeID(i, j, k, edge)];
		pTriIndices[index + 2] = mappededgeidx;		//notice: indices begin from 0.
		pTriVertex[mappededgeidx] = (vertlist[(edge*NTHREADS) + threadIdx.x]);
	}
}

__global__ void markActiveEdge_MC(uint *outmark, uint *compactedVoxelArray, farray volume, float isoValue, uint activeVoxels)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint idx = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (idx > activeVoxels - 1) {
		idx = activeVoxels - 1;
	}

	int voxel = compactedVoxelArray[idx];

	// compute position in 3d grid
	int i, j, k;
	getijk(i, j, k, voxel, NXMC, NYMC, NZMC);

	float field[8];
	field[0] = volume(i, j, k);
	field[1] = volume(i + 1, j, k);
	field[2] = volume(i + 1, j + 1, k);
	field[3] = volume(i, j + 1, k);
	field[4] = volume(i, j, k + 1);
	field[5] = volume(i + 1, j, k + 1);
	field[6] = volume(i + 1, j + 1, k + 1);
	field[7] = volume(i, j + 1, k + 1);

	// recalculate flag
	uint cubeindex;
	cubeindex = uint(field[0] < isoValue);
	cubeindex += uint(field[1] < isoValue) * 2;
	cubeindex += uint(field[2] < isoValue) * 4;
	cubeindex += uint(field[3] < isoValue) * 8;
	cubeindex += uint(field[4] < isoValue) * 16;
	cubeindex += uint(field[5] < isoValue) * 32;
	cubeindex += uint(field[6] < isoValue) * 64;
	cubeindex += uint(field[7] < isoValue) * 128;

	// output triangle vertices
	uint numVerts = tex1Dfetch(numVertsTex, cubeindex);
	uint edge;
	for (int idxVert = 0; idxVert<numVerts; idxVert++) {
		//下面可能会重复写，但是应该没问题。注意这个函数执行前需要把outmark置0
		edge = tex1Dfetch(triTex, (cubeindex * 16) + idxVert);
		outmark[GetEdgeID(i, j, k, edge)] = 1;
	}
	//debug
	// 	for( int edge=0; edge<12; edge++ )
	// 		outmark[GetEdgeID(i,j,k,edge)] = 1;

}

//以三角形为核心来计算法线，原子写入到点的法线中。注意：法线不要归一化
__global__ void calnormal_k(float3 *ppos, float3 *pnor, int pnum, uint *indices, int indicesnum)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < indicesnum / 3)		//face number
	{
		int i1 = indices[idx * 3 + 0];
		int i2 = indices[idx * 3 + 1];
		int i3 = indices[idx * 3 + 2];
		float3 p1 = ppos[i1];
		float3 p2 = ppos[i2];
		float3 p3 = ppos[i3];

		//compute
		float3 nor = cross(p2 - p1, p3 - p1);

		//write back
		atomicAdd(&pnor[i1].x, nor.x);
		atomicAdd(&pnor[i2].x, nor.x);
		atomicAdd(&pnor[i3].x, nor.x);
		atomicAdd(&pnor[i1].y, nor.y);
		atomicAdd(&pnor[i2].y, nor.y);
		atomicAdd(&pnor[i3].y, nor.y);
		atomicAdd(&pnor[i1].z, nor.z);
		atomicAdd(&pnor[i2].z, nor.z);
		atomicAdd(&pnor[i3].z, nor.z);
	}
}

//归一化顶点法线
__global__ void normalizeTriangleNor_k(float3 *pnor, int pnum)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < pnum)		//vertex number
	{
		if (length(pnor[idx])>0)
			pnor[idx] = normalize(pnor[idx]);
	}
}

void allocateTextures(uint **d_edgeTable, uint **d_triTable, uint **d_numVertsTable)
{
	checkCudaErrors(cudaMalloc((void**)d_edgeTable, 256 * sizeof(uint)));
	checkCudaErrors(cudaMemcpy((void *)*d_edgeTable, (void *)edgeTable, 256 * sizeof(uint), cudaMemcpyHostToDevice));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	checkCudaErrors(cudaBindTexture(0, edgeTex, *d_edgeTable, channelDesc));

	checkCudaErrors(cudaMalloc((void**)d_triTable, 256 * 16 * sizeof(uint)));
	checkCudaErrors(cudaMemcpy((void *)*d_triTable, (void *)triTable, 256 * 16 * sizeof(uint), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, triTex, *d_triTable, channelDesc));

	checkCudaErrors(cudaMalloc((void**)d_numVertsTable, 256 * sizeof(uint)));
	checkCudaErrors(cudaMemcpy((void *)*d_numVertsTable, (void *)numVertsTable, 256 * sizeof(uint), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, numVertsTex, *d_numVertsTable, channelDesc));
}

//计算两个1*n向量的点积，输出到out里(注意用归约求和的思想，out是一个数组，需要在CPU上累加起来)
__global__ void arrayproduct_k(float* out, float* x, float *y, int n)
{
	extern __shared__ float sdata[];
	uint tid = threadIdx.x;
	uint i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;

	sdata[tid] = (i >= n) ? 0 : (x[i] * y[i]);
	__syncthreads();

	for (int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid<s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}

	if (tid == 0)
		out[blockIdx.x] = sdata[0];
}

//z = Ax: A is a sparse matrix, representing the left hand item of Poisson equation.
__global__ void computeAx(farray ans, charray mark, farray x, int n)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<n)
	{
		if (mark[idx] == TYPEFLUID)		//todo: should add typesolid or not.
		{
			int i, j, k;
			getijk(i, j, k, idx);
			float center = x[idx];
			float sum = -6.0f*center;
			float h2_rev = dparam.cellsize.x*dparam.cellsize.x;
			//notice: x必须在AIR类型的格子里是0，下面的式子才正确
			sum += (mark(i + 1, j, k) == TYPEBOUNDARY) ? center : x(i + 1, j, k);
			sum += (mark(i, j + 1, k) == TYPEBOUNDARY) ? center : x(i, j + 1, k);
			sum += (mark(i, j, k + 1) == TYPEBOUNDARY) ? center : x(i, j, k + 1);
			sum += (mark(i - 1, j, k) == TYPEBOUNDARY) ? center : x(i - 1, j, k);
			sum += (mark(i, j - 1, k) == TYPEBOUNDARY) ? center : x(i, j - 1, k);
			sum += (mark(i, j, k - 1) == TYPEBOUNDARY) ? center : x(i, j, k - 1);
			ans[idx] = sum / h2_rev;
		}
		else
			ans[idx] = 0.0f;
	}
}

//Ans = x + a*y
__global__ void pcg_op(charray A, farray ans, farray x, farray y, float a, int n)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<n)
	{
		if (A[idx] == TYPEFLUID)
			ans[idx] = x[idx] + a*y[idx];
		else
			ans[idx] = 0.0f;
	}
}

__global__ void buildprecondition_pcg(farray P, charray mark, farray ans, farray input, int n)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<n)
	{
		ans[idx] = 1.0f / 6 * input[idx];
	}
}

__global__ void copyParticle2GL_vel_k(float3* ppos, float3 *pvel, float *pmass, char *pflag, int pnum, float *renderpos, float *rendercolor)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		renderpos[idx * 3] = ppos[idx].x;
		renderpos[idx * 3 + 1] = ppos[idx].y;
		renderpos[idx * 3 + 2] = ppos[idx].z;

		if (pflag[idx] == TYPEFLUID)
		{
			rendercolor[idx * 3] = 1.0f;
			rendercolor[idx * 3 + 1] = 0.0f;
			rendercolor[idx * 3 + 2] = 0.0f;
		}
		else if (pflag[idx] == TYPEAIR)
		{
			rendercolor[idx * 3] = 0.0f;
			rendercolor[idx * 3 + 1] = 0.0f;
			rendercolor[idx * 3 + 2] = 1.0f;
		}
		else if (pflag[idx] == TYPESOLID)
		{
			rendercolor[idx * 3] = 0.0f;
			rendercolor[idx * 3 + 1] = 1.0f;
			rendercolor[idx * 3 + 2] = 0.0f;
		}
	}
}

__global__ void copyParticle2GL_radius_k(float3* ppos, float *pmass, char *pflag, int pnum, float *renderpos, float *rendercolor, float minmass)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		renderpos[idx * 3] = ppos[idx].x;
		renderpos[idx * 3 + 1] = ppos[idx].y;
		renderpos[idx * 3 + 2] = ppos[idx].z;

		minmass *= 1.2f;		//trick
		float rate = (pmass[idx] - minmass*dparam.m0) / (dparam.m0 - minmass*dparam.m0);
		rate = fmax(0.0f, fmin(1.0f, rate));
		{
			float3 color = mapColorBlue2Red(powf(rate, 1.0f / 3)*6.0f);
			rendercolor[idx * 3] = color.x;
			rendercolor[idx * 3 + 1] = color.y;
			rendercolor[idx * 3 + 2] = color.z;
		}
	}
}



__device__ inline void atomicaddfloat3(float3 *a, int idx, float3 b)
{
	atomicAdd(&a[idx].x, b.x);
	atomicAdd(&a[idx].y, b.y);
	atomicAdd(&a[idx].z, b.z);
}

__global__ void smooth_computedisplacement(float3 *displacement, int *weight, float3 *ppos, uint *indices, int trianglenum)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<trianglenum)
	{
		uint p1 = indices[idx * 3];
		uint p2 = indices[idx * 3 + 1];
		uint p3 = indices[idx * 3 + 2];

		atomicaddfloat3(displacement, p1, ppos[p2] - ppos[p1]);
		atomicaddfloat3(displacement, p1, ppos[p3] - ppos[p1]);
		atomicaddfloat3(displacement, p2, ppos[p1] - ppos[p2]);
		atomicaddfloat3(displacement, p2, ppos[p3] - ppos[p2]);
		atomicaddfloat3(displacement, p3, ppos[p1] - ppos[p3]);
		atomicaddfloat3(displacement, p3, ppos[p2] - ppos[p3]);
		atomicAdd(&weight[p1], 2);
		atomicAdd(&weight[p2], 2);
		atomicAdd(&weight[p3], 2);
	}
}

__global__ void smooth_addDisplacement(float3 *displacement, int *weight, float3 *ppos, int vertexnum, float param)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<vertexnum)
	{
		if (weight[idx]>0)
			ppos[idx] += param * displacement[idx] / weight[idx];
		displacement[idx] = make_float3(0.0f);
		weight[idx] = 0;
	}
}

//diffuse density field.
__global__ void diffuse_dense(farray outp, farray inp, charray mark, float alpha, float beta)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < outp.xn * outp.yn * outp.zn)
	{
		float resp = 0;
		float p1, p2, p3, p4, p5, p6;
		float p0 = inp[idx];
		int i, j, k;
		getijk(i, j, k, idx, outp.xn, outp.yn, outp.zn);
		if (mark(i, j, k) == TYPEBOUNDARY)
			outp[idx] = 0.0f;
		else
		{
			p1 = (mark(i + 1, j, k) == TYPEBOUNDARY) ? p0 : inp(i + 1, j, k);
			p2 = (mark(i, j + 1, k) == TYPEBOUNDARY) ? p0 : inp(i, j + 1, k);
			p3 = (mark(i, j, k + 1) == TYPEBOUNDARY) ? p0 : inp(i, j, k + 1);
			p4 = (mark(i - 1, j, k) == TYPEBOUNDARY) ? p0 : inp(i - 1, j, k);
			p5 = (mark(i, j - 1, k) == TYPEBOUNDARY) ? p0 : inp(i, j - 1, k);
			p6 = (mark(i, j, k - 1) == TYPEBOUNDARY) ? p0 : inp(i, j, k - 1);
			resp = (p1 + p2 + p3 + p4 + p5 + p6 + alpha*p0) / beta;
			outp[idx] = resp;
		}
	}
}

//diffuse velocity field.
__global__ void diffuse_velocity(farray outv, farray inv, float alpha, float beta)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < outv.xn * outv.yn * outv.zn)
	{
		float resp = 0;
		float p1, p2, p3, p4, p5, p6;
		float p0 = inv[idx];
		int i, j, k;
		getijk(i, j, k, idx, outv.xn, outv.yn, outv.zn);
		if (i == 0 || j == 0 || k == 0 || i >= outv.xn - 1 || j >= outv.yn - 1 || k >= outv.zn - 1)
			outv[idx] = p0;
		else
		{
			p1 = inv(i + 1, j, k);
			p2 = inv(i, j + 1, k);
			p3 = inv(i, j, k + 1);
			p4 = inv(i - 1, j, k);
			p5 = inv(i, j - 1, k);
			p6 = inv(i, j, k - 1);
			resp = (p1 + p2 + p3 + p4 + p5 + p6 + alpha*p0) / beta;
			outv[idx] = resp;
		}
	}
}

//maxLength, hashPoints是输出：最长边（每个block里），每个三角形一个用来hash的点
__global__ void createAABB_q(float3* points, int nPoints, uint3* faces, int nFaces, float *maxLength, float3* hashPoints)
{
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= nFaces)
		return;

	__shared__ float maxArray[256];

	uint p1 = faces[index].x;
	uint p2 = faces[index].y;
	uint p3 = faces[index].z;

	//得到三角形的三个顶点
	float3 px = points[p1];
	float3 py = points[p2];
	float3 pz = points[p3];

	AABB aabb;
	aabb.xMin = (px.x>py.x) ? py.x : px.x;
	aabb.xMin = (aabb.xMin>pz.x) ? pz.x : aabb.xMin;
	aabb.xMax = (px.x<py.x) ? py.x : px.x;
	aabb.xMax = (aabb.xMax<pz.x) ? pz.x : aabb.xMax;

	aabb.yMin = (px.y>py.y) ? py.y : px.y;
	aabb.yMin = (aabb.yMin>pz.y) ? pz.y : aabb.yMin;
	aabb.yMax = (px.y<py.y) ? py.y : px.y;
	aabb.yMax = (aabb.yMax<pz.y) ? pz.y : aabb.yMax;

	aabb.zMin = (px.z>py.z) ? py.z : px.z;
	aabb.zMin = (aabb.zMin>pz.z) ? pz.z : aabb.zMin;
	aabb.zMax = (px.z<py.z) ? py.z : px.z;
	aabb.zMax = (aabb.zMax<pz.z) ? pz.z : aabb.zMax;

	float tempMaxLength = aabb.xMax - aabb.xMin;
	tempMaxLength = (tempMaxLength>aabb.yMax - aabb.yMin) ? (tempMaxLength) : (aabb.yMax - aabb.yMin);
	tempMaxLength = (tempMaxLength>aabb.zMax - aabb.zMin) ? (tempMaxLength) : (aabb.zMax - aabb.zMin);

	maxArray[threadIdx.x] = tempMaxLength;
	hashPoints[index] = make_float3((aabb.xMin + aabb.xMax) / 2, (aabb.yMin + aabb.yMax) / 2, (aabb.zMin + aabb.zMax) / 2);

	__syncthreads();

	for (int i = blockDim.x / 2; i>0; i /= 2)
	{
		if (threadIdx.x < i)
			maxArray[threadIdx.x] = max(maxArray[threadIdx.x], maxArray[i + threadIdx.x]);
		__syncthreads();
	}

	if (threadIdx.x == 0)
		maxLength[blockIdx.x] = maxArray[0];
}


__global__	void calcHash_radix_q(
	uint2*   gridParticleIndex, // output
	float3* posArray,               // input: positions
	uint    numParticles,
	float3 t_min,
	float3 t_max)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numParticles) return;

	float3 pos = posArray[index];
	uint hash;
	int gz = (pos.z - t_min.z) / dparam.triHashSize.z;
	int gy = (pos.y - t_min.y) / dparam.triHashSize.y;
	int gx = (pos.x - t_min.x) / dparam.triHashSize.x;
	if (gx < 0 || gx > dparam.triHashRes.x - 1 || gy < 0 || gy > dparam.triHashRes.y - 1 || gz < 0 || gz > dparam.triHashRes.z - 1)
		hash = CELL_UNDEF;
	else
		hash = __mul24(__mul24(gz, (int)dparam.triHashRes.y) + gy, (int)dparam.triHashRes.x) + gx;

	// store grid hash and particle index
	gridParticleIndex[index] = make_uint2(hash, index);
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStart_radix_q(uint*   cellStart,        // output: cell start index
uint*   cellEnd,          // output: cell end index
uint3* sortedFaces,
uint2 *  gridParticleHash, // input: sorted grid hashes
uint3* oldFaces,
uint    numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	uint hash;
	// handle case when no. of particles not multiple of block size
	if (index < numParticles) {
		hash = gridParticleHash[index].x;

		// Load hash data into shared memory so that we can look 
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index - 1].x;
		}
	}

	__syncthreads();

	if (index < numParticles) {
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
		uint sortedIndex = gridParticleHash[index].y;

		sortedFaces[index] = oldFaces[sortedIndex];       // see particles_kernel.cuh
	}
}

__global__ void calculateNormal(float3* points, uint3* faces, float3* normals, int num)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index < num)
	{
		uint3 face = faces[index];

		float3 v1 = points[face.x];
		float3 v2 = points[face.y];
		float3 v3 = points[face.z];

		float3 tmp;
		tmp.x = (v1.y - v2.y)*(v1.z - v3.z) - (v1.z - v2.z)*(v1.y - v3.y);
		tmp.y = (v1.z - v2.z)*(v1.x - v3.x) - (v1.x - v2.x)*(v1.z - v3.z);
		tmp.z = (v1.x - v2.x)*(v1.y - v3.y) - (v1.y - v2.y)*(v1.x - v3.x);

		normals[index] = normalize(tmp);
	}
}

//temp_yanglp: 检测一个小球与三角形是否相交，求出对粒子作用的顶点权重，返回值为负数，表示没有相交，正数表示相交
__device__ float IntersectTriangle_q(float3& pos, float radius, float3& v0, float3& v1, float3& v2, float3 n)
{
	//compute the distance of pos and triangle plane
	float d = dot(pos - v0, n);
	if (abs(d)>radius)		return -1;
	float dislimit = radius*radius - d*d;

	//球心在三角形平面的投影
	float3 pTri = pos - d*n;
	float3 tempcross;
	float d0 = dot(pTri - v0, pTri - v0);
	float d1 = dot(pTri - v1, pTri - v1);
	float d2 = dot(pTri - v2, pTri - v2);

	//判断是否在三角形内
	int tt = (dot(cross(pTri - v0, v1 - v0), n)>0) ? 1 : 0;
	tt += (dot(cross(pTri - v1, v2 - v1), n)>0) ? 2 : 0;
	tt += (dot(cross(pTri - v2, v0 - v2), n)>0) ? 4 : 0;
	//cuPrintf("tt=%d\n",tt);
	if (tt == 7 || tt == 0)
	{
		return abs(d);
	}

	//判断投影点与三角形顶点的距离是否符合条件 
	float distemp;
	float dis = (d0<dislimit) ? (d0) : dislimit;			//dis表示到目前为止投影点到三角形的最小距离

	dis = (d1<dis) ? (d1) : dis;

	dis = (d2<dis) ? (d2) : dis;

	//判断投影点与三角形边的距离
	if (dot(v1 - v0, pTri - v0)*dot(v0 - v1, pTri - v1)>0)
	{
		tempcross = cross(v1 - v0, pTri - v0);
		distemp = dot(tempcross, tempcross) / dot(v1 - v0, v1 - v0);
		dis = (distemp<dis) ? (distemp) : dis;
	}
	if (dot(v2 - v1, pTri - v1)*dot(v1 - v2, pTri - v2)>0)
	{
		tempcross = cross(v2 - v1, pTri - v1);
		distemp = dot(tempcross, tempcross) / dot(v2 - v1, v2 - v1);
		dis = (distemp<dis) ? (distemp) : dis;
	}
	if (dot(v0 - v2, pTri - v2)*dot(v2 - v0, pTri - v0)>0)
	{
		tempcross = cross(v0 - v2, pTri - v2);
		distemp = dot(tempcross, tempcross) / dot(v0 - v2, v0 - v2);
		dis = (distemp<dis) ? (distemp) : dis;
	}

	if (dis > dislimit - 0.001)	return -1;

	return sqrt(dis + d*d);
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash_q(int3 gridPos)
{
	return __umul24(__umul24(gridPos.z, dparam.triHashRes.y), dparam.triHashRes.x) + __umul24(gridPos.y, dparam.triHashRes.x) + gridPos.x;
}

// collide a particle against all other particles in a given cell
__device__
float3 collideCell(int3    gridPos,
float3  pos,
float radius,
float3* surPoints,
uint3* surIndex,
float3* surfaceNor,
uint*   cellStart,
uint*   cellEnd,
int scene)
{
	uint gridHash = calcGridHash_q(gridPos);

	float dis_n, wib = 0;
	float3 force = make_float3(0.0f);
	// get start of bucket for this cell
	uint startIndex = cellStart[gridHash];

	if (startIndex != CELL_UNDEF) {        // cell is not empty
		// iterate over particles in this cell
		uint endIndex = cellEnd[gridHash];
		for (uint j = startIndex; j<endIndex; j++) {
			//cuPrintf("j=%d\n", j);
			dis_n = IntersectTriangle_q(pos, radius, surPoints[surIndex[j].x], surPoints[surIndex[j].y], surPoints[surIndex[j].z], surfaceNor[j]);
			wib = 1 - dis_n / radius;
			if (dis_n >= 0 && wib > 0.00001)
			{
				force += (radius - dis_n) * (surfaceNor[j]) * 10;
			}
		}
	}

	return force;
}

__device__ void mindis_cell(float& mindisair, float& mindisfluid, float3 gpos, float3 *pos, char *parflag, float *pmass, uint *gridstart, uint  *gridend, int gidx)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis;
	for (uint p = start; p<end; ++p)
	{
		dis = length(pos[p] - gpos);// - getRfromMass( pmass[p] ) - 0.18f*dparam.cellsize.x;		//减掉半径，后面的数是较正一下
		if (parflag[p] == TYPEAIR || parflag[p] == TYPEAIRSOLO)//todo: 是不是加上SOLO的类型以防止ls随着标记变化的突变？
			mindisair = (dis<mindisair) ? dis : mindisair;
		else if (parflag[p] == TYPEFLUID || parflag[p] == TYPESOLID)
			mindisfluid = (dis<mindisfluid) ? dis : mindisfluid;
	}
}


//这个level set的值很可能有问题，从画出来的图可以看出来一些，直接影响后面所有的内容。
//[2012]【长文】MultiFLIP for Energetic Two-Phase Fluid Simulation
__global__ void genlevelset(farray lsfluid, farray lsair, charray mark, float3 *pos, char *parflag, float *pmass, uint *gridstart, uint  *gridend, float fMCDensity, float offset)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx < dparam.gnum)		//每个格子一个值
	{
		//float ls;
		float h = dparam.cellsize.x;
		mark[idx] = TYPEVACUUM;
		float r = 0.5f*h;		//0.36f*h;
		//get position
		int i, j, k;
		getijk(i, j, k, idx, NX, NY, NZ);

		float3 gpos = (make_float3(i, j, k) + make_float3(0.5f, 0.5f, 0.5f))*dparam.cellsize.x;
		float mindisair = 2.5f*h, mindisfluid = 2.5f*h;	//2.5 cellsize
		int level = 2;
		for (int di = -level; di <= level; ++di) for (int dj = -level; dj <= level; ++dj) for (int dk = -level; dk <= level; ++dk)	//周围27个格子就行
		{
			if (verifycellidx(i + di, j + dj, k + dk))
			{
				mindis_cell(mindisair, mindisfluid, gpos, pos, parflag, pmass, gridstart, gridend, getidx(i + di, j + dj, k + dk));
			}
		}
		mindisair -= r;
		mindisfluid -= r;

		lsfluid[idx] = mindisfluid;
	//	lsair[idx] = mindisair - offset*h;	//todo: 这里略微向外扩张了一下气体的ls，避免气体粒子correctpos时向内收缩导到气泡体积的减小。注意：这个修正会导致markgrid的不对，因此流体mark会大一层，其流动会受很大影响
		lsair[idx] = mindisair;
	}
}


__device__ void sumcell_fluidSolid(float3 &usum, float &weight, float3 gpos, float3 *pos, float3 *vel, float *mass, char *parflag, uint *gridstart, uint  *gridend, int gidx)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis2, w, RE = 1.4;
	float scale = 1 / dparam.cellsize.x;
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == TYPEFLUID || parflag[p] == TYPESOLID)
		{
			dis2 = dot(pos[p] * scale - gpos, pos[p] * scale - gpos);		//scale is necessary.
			w = mass[p] * sharp_kernel(dis2, RE);
			weight += w;
			usum += w*vel[p];
		}
	}
}

__global__ void mapvelp2g_k_fluidSolid(float3 *pos, float3 *vel, float *mass, char *parflag, int pnum, farray ux, farray uy, farray uz, uint* gridstart, uint *gridend)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float weight;
	float3 gpos, usum;
	if (idx<dparam.gvnum.x)
	{
		// ux
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		gpos.x = i, gpos.y = j + 0.5, gpos.z = k + 0.5;
		for (int di = -1; di <= 0; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_fluidSolid(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.x = (weight>0) ? (usum.x / weight) : 0.0f;
		ux(i, j, k) = usum.x;
	}
	if (idx<dparam.gvnum.y)
	{
		// uy
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		gpos.x = i + 0.5, gpos.y = j, gpos.z = k + 0.5;
		for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 0; dj++) for (int dk = -1; dk <= 1; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_fluidSolid(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.y = (weight>0) ? (usum.y / weight) : 0.0f;
		uy(i, j, k) = usum.y;
	}
	if (idx<dparam.gvnum.z)
	{
		// uz
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		gpos.x = i + 0.5, gpos.y = j + 0.5, gpos.z = k;
		for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 0; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_fluidSolid(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.z = (weight>0) ? (usum.z / weight) : 0.0f;
		uz(i, j, k) = usum.z;
	}
}

__device__ void sumcell_air(float3 &usum, float &weight, float3 gpos, float3 *pos, float3 *vel,
	float *mass, char *parflag, uint *gridstart, uint  *gridend, int gidx)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis2, w, RE = 1.4;
	float scale = 1 / dparam.cellsize.x;
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == TYPEAIR)
		{
			dis2 = dot(pos[p] * scale - gpos, pos[p] * scale - gpos);		//scale is necessary.
			w = mass[p] * sharp_kernel(dis2, RE);
			weight += w;
			usum += w*vel[p];
		}
	}
}

__global__ void mapvelp2g_k_air(float3 *pos, float3 *vel, float *mass, char *parflag, int pnum, farray ux, farray uy, farray uz, uint* gridstart, uint *gridend)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float weight;
	float3 gpos, usum;
	int rangemax = 2, rangemin = 1;
	if (idx<dparam.gvnum.x)
	{
		// ux
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		gpos.x = i, gpos.y = j + 0.5, gpos.z = k + 0.5;
		for (int di = -rangemax; di <= rangemin; di++) for (int dj = -rangemax; dj <= rangemax; dj++) for (int dk = -rangemax; dk <= rangemax; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_air(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.x = (weight>0) ? (usum.x / weight) : 0.0f;
		ux(i, j, k) = usum.x;
	}
	if (idx<dparam.gvnum.y)
	{
		// uy
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		gpos.x = i + 0.5, gpos.y = j, gpos.z = k + 0.5;
		for (int di = -rangemax; di <= rangemax; di++) for (int dj = -rangemax; dj <= rangemin; dj++) for (int dk = -rangemax; dk <= rangemax; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_air(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.y = (weight>0) ? (usum.y / weight) : 0.0f;
		uy(i, j, k) = usum.y;
	}
	if (idx<dparam.gvnum.z)
	{
		// uz
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		gpos.x = i + 0.5, gpos.y = j + 0.5, gpos.z = k;
		for (int di = -rangemax; di <= rangemax; di++) for (int dj = -rangemax; dj <= rangemax; dj++) for (int dk = -rangemax; dk <= rangemin; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_air(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.z = (weight>0) ? (usum.z / weight) : 0.0f;
		uz(i, j, k) = usum.z;
	}
}
__device__ void sumcell_solid(float3 &usum, float &weight, float3 gpos, float3 *pos, float3 *vel,
	float *mass, char *parflag, uint *gridstart, uint  *gridend, int gidx)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis2, w, RE = 1.4;
	float scale = 1 / dparam.cellsize.x;
	for (uint p = start; p<end; ++p)
	{
		if (parflag[p] == TYPESOLID)
		{
			dis2 = dot(pos[p] * scale - gpos, pos[p] * scale - gpos);		//scale is necessary.
			w = mass[p] * sharp_kernel(dis2, RE);
			weight += w;
			usum += w*vel[p];
		}
	}
}

__global__ void mapvelp2g_k_solid(float3 *pos, float3 *vel, float *mass, char *parflag, int pnum, farray ux, farray uy, farray uz, uint* gridstart, uint *gridend)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float weight;
	float3 gpos, usum;
	int rangemax = 2, rangemin = 1;
	if (idx<dparam.gvnum.x)
	{
		// ux
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		gpos.x = i, gpos.y = j + 0.5, gpos.z = k + 0.5;
		for (int di = -rangemax; di <= rangemin; di++) for (int dj = -rangemax; dj <= rangemax; dj++) for (int dk = -rangemax; dk <= rangemax; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_solid(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.x = (weight>0) ? (usum.x / weight) : 0.0f;
		ux(i, j, k) = usum.x;
	}
	if (idx<dparam.gvnum.y)
	{
		// uy
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		gpos.x = i + 0.5, gpos.y = j, gpos.z = k + 0.5;
		for (int di = -rangemax; di <= rangemax; di++) for (int dj = -rangemax; dj <= rangemin; dj++) for (int dk = -rangemax; dk <= rangemax; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_solid(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.y = (weight>0) ? (usum.y / weight) : 0.0f;
		uy(i, j, k) = usum.y;
	}
	if (idx<dparam.gvnum.z)
	{
		// uz
		weight = 0, usum = make_float3(0.0f);
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		gpos.x = i + 0.5, gpos.y = j + 0.5, gpos.z = k;
		for (int di = -rangemax; di <= rangemax; di++) for (int dj = -rangemax; dj <= rangemax; dj++) for (int dk = -rangemax; dk <= rangemin; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumcell_solid(usum, weight, gpos, pos, vel, mass, parflag, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		usum.z = (weight>0) ? (usum.z / weight) : 0.0f;
		uz(i, j, k) = usum.z;
	}
}

//计算散度
__global__ void cptdivergence_bubble(farray outdiv, farray waterux, farray wateruy, farray wateruz, farray airux, farray airuy, farray airuz, charray mark, farray ls, farray sf)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx <dparam.gnum)
	{
		float div = 0, h = dparam.cellsize.x;
		int i, j, k;
		getijk(i, j, k, idx);

		float ux0, ux1, uy0, uy1, uz0, uz1;
		float jx0, jx1, jy0, jy1, jz0, jz1, J;		//surface tension, [2005]Discontinuous Fluids
		float theta;
		if (mark[idx] == TYPEFLUID || mark[idx] == TYPEAIR)
		{
			//ux1
			if (mark[idx] == TYPEFLUID && mark(i + 1, j, k) != TYPEAIR)
				ux1 = waterux(i + 1, j, k), jx1 = 0;
			else if (mark[idx] == TYPEAIR && mark(i + 1, j, k) != TYPEFLUID)
				ux1 = airux(i + 1, j, k), jx1 = 0;
			else if (mark[idx] == TYPEFLUID && mark(i + 1, j, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i + 1, j, k) - ls(i, j, k));
				ux1 = theta * waterux(i + 1, j, k) + (1 - theta) * airux(i + 1, j, k);
				jx1 = theta * sf(i, j, k) + (1 - theta) * sf(i + 1, j, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i + 1, j, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i + 1, j, k) - ls(i, j, k));
				ux1 = theta * airux(i + 1, j, k) + (1 - theta) * waterux(i + 1, j, k);
				jx1 = theta * sf(i, j, k) + (1 - theta) * sf(i + 1, j, k);
			}

			//ux0
			if (mark[idx] == TYPEFLUID && mark(i - 1, j, k) != TYPEAIR)
				ux0 = waterux(i, j, k), jx0 = 0;
			else if (mark[idx] == TYPEAIR && mark(i - 1, j, k) != TYPEFLUID)
				ux0 = airux(i, j, k), jx0 = 0;
			else if (mark[idx] == TYPEFLUID && mark(i - 1, j, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i - 1, j, k) - ls(i, j, k));
				ux0 = theta * waterux(i, j, k) + (1 - theta) * airux(i, j, k);
				jx0 = theta*sf(i, j, k) + (1 - theta)*sf(i - 1, j, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i - 1, j, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i - 1, j, k) - ls(i, j, k));
				ux0 = theta * airux(i, j, k) + (1 - theta) * waterux(i, j, k);
				jx0 = theta*sf(i, j, k) + (1 - theta)*sf(i - 1, j, k);
			}

			//uy1
			if (mark[idx] == TYPEFLUID && mark(i, j + 1, k) != TYPEAIR)
				uy1 = wateruy(i, j + 1, k), jy1 = 0;
			else if (mark[idx] == TYPEAIR && mark(i, j + 1, k) != TYPEFLUID)
				uy1 = airuy(i, j + 1, k), jy1 = 0;
			else if (mark[idx] == TYPEFLUID && mark(i, j + 1, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j + 1, k) - ls(i, j, k));
				uy1 = theta * wateruy(i, j + 1, k) + (1 - theta) * airuy(i, j + 1, k);
				jy1 = theta*sf(i, j, k) + (1 - theta)*sf(i, j + 1, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j + 1, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j + 1, k) - ls(i, j, k));
				uy1 = theta * airuy(i, j + 1, k) + (1 - theta) * wateruy(i, j + 1, k);
				jy1 = theta*sf(i, j, k) + (1 - theta)*sf(i, j + 1, k);
			}

			//uy0
			if (mark[idx] == TYPEFLUID && mark(i, j - 1, k) != TYPEAIR)
				uy0 = wateruy(i, j, k), jy0 = 0;
			else if (mark[idx] == TYPEAIR && mark(i, j - 1, k) != TYPEFLUID)
				uy0 = airuy(i, j, k), jy0 = 0;
			else if (mark[idx] == TYPEFLUID && mark(i, j - 1, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j - 1, k) - ls(i, j, k));
				uy0 = theta * wateruy(i, j, k) + (1 - theta) * airuy(i, j, k);
				jy0 = theta*sf(i, j, k) + (1 - theta)*sf(i, j - 1, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j - 1, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j - 1, k) - ls(i, j, k));
				uy0 = theta * airuy(i, j, k) + (1 - theta) * wateruy(i, j, k);
				jy0 = theta*sf(i, j, k) + (1 - theta)*sf(i, j - 1, k);
			}

			//uz1
			if (mark[idx] == TYPEFLUID && mark(i, j, k + 1) != TYPEAIR)
				uz1 = wateruz(i, j, k + 1), jz1 = 0;
			else if (mark[idx] == TYPEAIR && mark(i, j, k + 1) != TYPEFLUID)
				uz1 = airuz(i, j, k + 1), jz1 = 0;
			else if (mark[idx] == TYPEFLUID && mark(i, j, k + 1) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k + 1) - ls(i, j, k));
				uz1 = theta * wateruz(i, j, k + 1) + (1 - theta) * airuz(i, j, k + 1);
				jz1 = theta*sf(i, j, k) + (1 - theta)*sf(i, j, k + 1);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j, k + 1) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k + 1) - ls(i, j, k));
				uz1 = theta * airuz(i, j, k + 1) + (1 - theta) * wateruz(i, j, k + 1);
				jz1 = theta*sf(i, j, k) + (1 - theta)*sf(i, j, k + 1);
			}

			//uz0
			if (mark[idx] == TYPEFLUID && mark(i, j, k - 1) != TYPEAIR)
				uz0 = wateruz(i, j, k), jz0 = 0;
			else if (mark[idx] == TYPEAIR && mark(i, j, k - 1) != TYPEFLUID)
				uz0 = airuz(i, j, k), jz0 = 0;
			else if (mark[idx] == TYPEFLUID && mark(i, j, k - 1) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k - 1) - ls(i, j, k));
				uz0 = theta * wateruz(i, j, k) + (1 - theta) * airuz(i, j, k);
				jz0 = theta*sf(i, j, k) + (1 - theta)*sf(i, j, k - 1);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j, k - 1) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k - 1) - ls(i, j, k));
				uz0 = theta * airuz(i, j, k) + (1 - theta) * wateruz(i, j, k);
				jz0 = theta*sf(i, j, k) + (1 - theta)*sf(i, j, k - 1);
			}

			J = (jx1 - jx0 + jy1 - jy0 + jz1 - jz0) / h / h;

			div = (ux1 - ux0 + uy1 - uy0 + uz1 - uz0) / h;
			div += J;	//surfacetension
		}

		outdiv[idx] = div;
	}
}

//计算散度，不使用压强来施加表面张力
__global__ void cptdivergence_bubble2(farray outdiv, farray waterux, farray wateruy, farray wateruz, farray airux, farray airuy, farray airuz, charray mark, farray ls)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx <dparam.gnum)
	{
		float div = 0, h = dparam.cellsize.x;
		int i, j, k;
		getijk(i, j, k, idx);

		float ux0, ux1, uy0, uy1, uz0, uz1;
		float theta;
		if (mark[idx] == TYPEFLUID || mark[idx] == TYPEAIR)
		{
			//ux1
			if (mark[idx] == TYPEFLUID && mark(i + 1, j, k) != TYPEAIR)
				ux1 = waterux(i + 1, j, k);
			else if (mark[idx] == TYPEAIR && mark(i + 1, j, k) != TYPEFLUID)
				ux1 = airux(i + 1, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i + 1, j, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i + 1, j, k) - ls(i, j, k));
				ux1 = theta * waterux(i + 1, j, k) + (1 - theta) * airux(i + 1, j, k);
				//ux1 = airux(i+1,j,k);
			}
			else if (mark[idx] == TYPEAIR && mark(i + 1, j, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i + 1, j, k) - ls(i, j, k));
				ux1 = theta * airux(i + 1, j, k) + (1 - theta) * waterux(i + 1, j, k);
				//ux1 = airux(i+1,j,k);
			}

			//ux0
			if (mark[idx] == TYPEFLUID && mark(i - 1, j, k) != TYPEAIR)
				ux0 = waterux(i, j, k);
			else if (mark[idx] == TYPEAIR && mark(i - 1, j, k) != TYPEFLUID)
				ux0 = airux(i, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i - 1, j, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i - 1, j, k) - ls(i, j, k));
				ux0 = theta * waterux(i, j, k) + (1 - theta) * airux(i, j, k);
				//ux0 = airux(i,j,k);
			}
			else if (mark[idx] == TYPEAIR && mark(i - 1, j, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i - 1, j, k) - ls(i, j, k));
				ux0 = theta * airux(i, j, k) + (1 - theta) * waterux(i, j, k);
				//ux0 = airux(i,j,k);
			}

			//uy1
			if (mark[idx] == TYPEFLUID && mark(i, j + 1, k) != TYPEAIR)
				uy1 = wateruy(i, j + 1, k);
			else if (mark[idx] == TYPEAIR && mark(i, j + 1, k) != TYPEFLUID)
				uy1 = airuy(i, j + 1, k);
			else if (mark[idx] == TYPEFLUID && mark(i, j + 1, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j + 1, k) - ls(i, j, k));
				uy1 = theta * wateruy(i, j + 1, k) + (1 - theta) * airuy(i, j + 1, k);
				//uy1 = airuy(i,j+1,k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j + 1, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j + 1, k) - ls(i, j, k));
				uy1 = theta * airuy(i, j + 1, k) + (1 - theta) * wateruy(i, j + 1, k);
				//uy1 = airuy(i,j+1,k);
			}

			//uy0
			if (mark[idx] == TYPEFLUID && mark(i, j - 1, k) != TYPEAIR)
				uy0 = wateruy(i, j, k);
			else if (mark[idx] == TYPEAIR && mark(i, j - 1, k) != TYPEFLUID)
				uy0 = airuy(i, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i, j - 1, k) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j - 1, k) - ls(i, j, k));
				uy0 = theta * wateruy(i, j, k) + (1 - theta) * airuy(i, j, k);
				//	uy0 = airuy(i,j,k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j - 1, k) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j - 1, k) - ls(i, j, k));
				uy0 = theta * airuy(i, j, k) + (1 - theta) * wateruy(i, j, k);
				//uy0 = airuy(i,j,k);
			}

			//uz1
			if (mark[idx] == TYPEFLUID && mark(i, j, k + 1) != TYPEAIR)
				uz1 = wateruz(i, j, k + 1);
			else if (mark[idx] == TYPEAIR && mark(i, j, k + 1) != TYPEFLUID)
				uz1 = airuz(i, j, k + 1);
			else if (mark[idx] == TYPEFLUID && mark(i, j, k + 1) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k + 1) - ls(i, j, k));
				uz1 = theta * wateruz(i, j, k + 1) + (1 - theta) * airuz(i, j, k + 1);
				//uz1 = airuz(i,j,k+1);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j, k + 1) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k + 1) - ls(i, j, k));
				uz1 = theta * airuz(i, j, k + 1) + (1 - theta) * wateruz(i, j, k + 1);
				//uz1 = airuz(i,j,k+1);
			}

			//uz0
			if (mark[idx] == TYPEFLUID && mark(i, j, k - 1) != TYPEAIR)
				uz0 = wateruz(i, j, k);
			else if (mark[idx] == TYPEAIR && mark(i, j, k - 1) != TYPEFLUID)
				uz0 = airuz(i, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i, j, k - 1) == TYPEAIR)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k - 1) - ls(i, j, k));
				uz0 = theta * wateruz(i, j, k) + (1 - theta) * airuz(i, j, k);
				//uz0 = airuz(i,j,k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j, k - 1) == TYPEFLUID)
			{
				theta = (0.0f - ls(i, j, k)) / (ls(i, j, k - 1) - ls(i, j, k));
				uz0 = theta * airuz(i, j, k) + (1 - theta) * wateruz(i, j, k);
				//uz0 = airuz(i,j,k);
			}
			div = (ux1 - ux0 + uy1 - uy0 + uz1 - uz0) / h;
		}

		outdiv[idx] = div;
	}
}

__global__ void cptdivergence_bubble3(farray outdiv, farray waterux, farray wateruy, farray wateruz, farray airux, farray airuy, farray airuz, charray mark, farray ls)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx <dparam.gnum)
	{
		float div = 0, h = dparam.cellsize.x;
		int i, j, k;
		getijk(i, j, k, idx);

		float ux0, ux1, uy0, uy1, uz0, uz1;
		float theta;
		if (mark[idx] == TYPEFLUID || mark[idx] == TYPEAIR)
		{
			//ux1
			if (mark[idx] == TYPEFLUID && mark(i + 1, j, k) != TYPEAIR)
				ux1 = waterux(i + 1, j, k);
			else if (mark[idx] == TYPEAIR && mark(i + 1, j, k) != TYPEFLUID)
				ux1 = airux(i + 1, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i + 1, j, k) == TYPEAIR)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i+1,j,k)-ls(i,j,k));
				// 				ux1 = theta * waterux(i+1,j,k) + (1-theta) * airux(i+1,j,k);
				ux1 = airux(i + 1, j, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i + 1, j, k) == TYPEFLUID)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i+1,j,k)-ls(i,j,k));
				// 				ux1 = theta * airux(i+1,j,k) + (1-theta) * waterux(i+1,j,k);
				ux1 = airux(i + 1, j, k);
			}

			//ux0
			if (mark[idx] == TYPEFLUID && mark(i - 1, j, k) != TYPEAIR)
				ux0 = waterux(i, j, k);
			else if (mark[idx] == TYPEAIR && mark(i - 1, j, k) != TYPEFLUID)
				ux0 = airux(i, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i - 1, j, k) == TYPEAIR)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i-1,j,k)-ls(i,j,k));
				// 				ux0 = theta * waterux(i,j,k) + (1-theta) * airux(i,j,k);
				ux0 = airux(i, j, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i - 1, j, k) == TYPEFLUID)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i-1,j,k)-ls(i,j,k));
				// 				ux0 = theta * airux(i,j,k) + (1-theta) * waterux(i,j,k);
				ux0 = airux(i, j, k);
			}

			//uy1
			if (mark[idx] == TYPEFLUID && mark(i, j + 1, k) != TYPEAIR)
				uy1 = wateruy(i, j + 1, k);
			else if (mark[idx] == TYPEAIR && mark(i, j + 1, k) != TYPEFLUID)
				uy1 = airuy(i, j + 1, k);
			else if (mark[idx] == TYPEFLUID && mark(i, j + 1, k) == TYPEAIR)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i,j+1,k)-ls(i,j,k));
				// 				uy1 = theta * wateruy(i,j+1,k) + (1-theta) * airuy(i,j+1,k);
				uy1 = airuy(i, j + 1, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j + 1, k) == TYPEFLUID)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i,j+1,k)-ls(i,j,k));
				// 				uy1 = theta * airuy(i,j+1,k) + (1-theta) * wateruy(i,j+1,k);
				uy1 = airuy(i, j + 1, k);
			}

			//uy0
			if (mark[idx] == TYPEFLUID && mark(i, j - 1, k) != TYPEAIR)
				uy0 = wateruy(i, j, k);
			else if (mark[idx] == TYPEAIR && mark(i, j - 1, k) != TYPEFLUID)
				uy0 = airuy(i, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i, j - 1, k) == TYPEAIR)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i,j-1,k)-ls(i,j,k));
				// 				uy0 = theta * wateruy(i,j,k) + (1-theta) * airuy(i,j,k);
				uy0 = airuy(i, j, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j - 1, k) == TYPEFLUID)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i,j-1,k)-ls(i,j,k));
				// 				uy0 = theta * airuy(i,j,k) + (1-theta) * wateruy(i,j,k);
				uy0 = airuy(i, j, k);
			}

			//uz1
			if (mark[idx] == TYPEFLUID && mark(i, j, k + 1) != TYPEAIR)
				uz1 = wateruz(i, j, k + 1);
			else if (mark[idx] == TYPEAIR && mark(i, j, k + 1) != TYPEFLUID)
				uz1 = airuz(i, j, k + 1);
			else if (mark[idx] == TYPEFLUID && mark(i, j, k + 1) == TYPEAIR)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i,j,k+1)-ls(i,j,k));
				// 				uz1 = theta * wateruz(i,j,k+1) + (1-theta) * airuz(i,j,k+1);
				uz1 = airuz(i, j, k + 1);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j, k + 1) == TYPEFLUID)
			{
				// 				theta = (0.0f-ls(i,j,k))/(ls(i,j,k+1)-ls(i,j,k));
				// 				uz1 = theta * airuz(i,j,k+1) + (1-theta) * wateruz(i,j,k+1);
				uz1 = airuz(i, j, k + 1);
			}

			//uz0
			if (mark[idx] == TYPEFLUID && mark(i, j, k - 1) != TYPEAIR)
				uz0 = wateruz(i, j, k);
			else if (mark[idx] == TYPEAIR && mark(i, j, k - 1) != TYPEFLUID)
				uz0 = airuz(i, j, k);
			else if (mark[idx] == TYPEFLUID && mark(i, j, k - 1) == TYPEAIR)
			{
				// 				theta=(0.0f-ls(i,j,k))/(ls(i,j,k-1)-ls(i,j,k));
				// 				uz0 = theta * wateruz(i,j,k) + (1-theta) * airuz(i,j,k);
				uz0 = airuz(i, j, k);
			}
			else if (mark[idx] == TYPEAIR && mark(i, j, k - 1) == TYPEFLUID)
			{
				// 				theta=(0.0f-ls(i,j,k))/(ls(i,j,k-1)-ls(i,j,k));
				// 				uz0 = theta * airuz(i,j,k) + (1-theta) * wateruz(i,j,k);
				uz0 = airuz(i, j, k);
			}

			div = (ux1 - ux0 + uy1 - uy0 + uz1 - uz0) / h;
		}

		outdiv[idx] = div;
	}
}

//压强与速度的计算
__global__ void subGradPress_bubble(farray p, farray ux, farray uy, farray uz)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float h = dparam.cellsize.x;
	if (idx<dparam.gvnum.x)
	{
		//ux
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		if (i>0 && i<NX)		//look out for this condition
			ux(i, j, k) -= (p(i, j, k) - p(i - 1, j, k)) / h;
	}
	if (idx<dparam.gvnum.y)
	{
		//uy
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		if (j>0 && j<NY)		//look out for this condition
			uy(i, j, k) -= (p(i, j, k) - p(i, j - 1, k)) / h;
	}
	if (idx<dparam.gvnum.z)
	{
		//uz
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		if (k>0 && k<NZ)		//look out for this condition
			uz(i, j, k) -= (p(i, j, k) - p(i, j, k - 1)) / h;
	}
}

//z = Ax: A is a sparse matrix, representing the left hand item of Poisson equation.
__global__ void computeAx_bubble(farray ans, charray mark, farray x, int n)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<n)
	{
		if (mark[idx] == TYPEFLUID || mark[idx] == TYPEAIR)
		{
			int i, j, k;
			getijk(i, j, k, idx);
			float center = x[idx];
			float sum = -6.0f*center;
			float h2_rev = dparam.cellsize.x*dparam.cellsize.x;

			sum += (mark(i + 1, j, k) == TYPEBOUNDARY) ? center : x(i + 1, j, k);
			sum += (mark(i, j + 1, k) == TYPEBOUNDARY) ? center : x(i, j + 1, k);
			sum += (mark(i, j, k + 1) == TYPEBOUNDARY) ? center : x(i, j, k + 1);
			sum += (mark(i - 1, j, k) == TYPEBOUNDARY) ? center : x(i - 1, j, k);
			sum += (mark(i, j - 1, k) == TYPEBOUNDARY) ? center : x(i, j - 1, k);
			sum += (mark(i, j, k - 1) == TYPEBOUNDARY) ? center : x(i, j, k - 1);
			ans[idx] = sum / h2_rev;
		}
		else
			ans[idx] = 0.0f;
	}
}

//Ans = x + a*y
__global__ void pcg_op_bubble(charray A, farray ans, farray x, farray y, float a, int n)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<n)
	{
		if (A[idx] == TYPEFLUID || A[idx] == TYPEAIR)
			ans[idx] = x[idx] + a*y[idx];
		else
			ans[idx] = 0.0f;
	}
}

//注意：这个函数只更新流体粒子(TYPEFLUID)的位置，但更新AIR粒子的速度(不是AIRSOLO)（用CIP模式）.
__global__ void advectparticle_RK2_bubble(float3 *ppos, float3 *pvel, int pnum, farray waterux, farray wateruy, farray wateruz,
	farray airux, farray airuy, farray airuz, float dt, char *parflag, VELOCITYMODEL velmode)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (parflag[idx] == TYPEAIRSOLO)		//对于小的气体粒子AIRSOLO，什么也不更新，跳过
			return;

		//read in
		float3 ipos = ppos[idx], ivel = pvel[idx];
		float3 tmin = dparam.gmin + (dparam.cellsize + make_float3(0.5f*dparam.cellsize.x));
		float3 tmax = dparam.gmax - (dparam.cellsize + make_float3(0.5f*dparam.cellsize.x));
		char partype = parflag[idx];

		//pos-->grid xyz
		float3 gvel = make_float3(0.0f);
		if (partype == TYPEFLUID)
			gvel = getParticleVelFromGrid(ipos, waterux, wateruy, wateruz);
		else if (partype == TYPEAIR)
			gvel = getParticleVelFromGrid(ipos, airux, airuy, airuz);
		else		//TYPEAIRSOLO 有自己的仿真方法，不参与这些仿真
			return;

		if (velmode == CIP /*|| partype==TYPEAIR*/)		//todo: 气体粒子用cip模式，减少乱跑的可能
			ivel = gvel;
		else
			ivel = (1 - FLIP_ALPHA)*gvel + FLIP_ALPHA*pvel[idx];

		//mid point: x(n+1/2) = x(n) + 0.5*dt*u(xn)
		float3 midpoint = ipos + gvel * dt * 0.5;
		float3 gvelmidpoint;
		if (partype == TYPEFLUID)
			gvelmidpoint = getParticleVelFromGrid(midpoint, waterux, wateruy, wateruz);
		else
			gvelmidpoint = getParticleVelFromGrid(midpoint, airux, airuy, airuz);

		// x(n+1) = x(n) + dt*u(x+1/2)
		ipos += gvelmidpoint * dt;

		//check boundary
		if (ipos.x <= tmin.x)
			ipos.x = tmin.x, ivel.x = 0.0f;
		if (ipos.y <= tmin.y)
			ipos.y = tmin.y, ivel.y = 0.0f;
		if (ipos.z <= tmin.z)
			ipos.z = tmin.z, ivel.z = 0.0f;

		if (ipos.x >= tmax.x)
			ipos.x = tmax.x, ivel.x = 0.0f;
		if (ipos.y >= tmax.y)
			ipos.y = tmax.y, ivel.y = 0.0f;
		if (ipos.z >= tmax.z)
			ipos.z = tmax.z, ivel.z = 0.0f;

		//write back: TYPEAIR+TYPESOLID只更新速度，TYPESOLO之前已经return，TYPEFLUID更新位置和速度。
		pvel[idx] = ivel;
		// 		if( partype==TYPEFLUID )
		// 			ppos[idx] = ipos;
	}
}

__global__ void mapvelg2p_flip_bubble(float3 *ppos, float3 *vel, char* parflag, int pnum, farray waterux, farray wateruy, farray wateruz, farray airux, farray airuy, farray airuz)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//pos-->grid xyz
		float3 ipos = ppos[idx];
		float3 gvel = make_float3(0.0f);
		if (parflag[idx] == TYPEFLUID || parflag[idx] == TYPESOLID)
			gvel = getParticleVelFromGrid(ipos, waterux, wateruy, wateruz);
		else if (parflag[idx] == TYPEAIR)
			gvel = getParticleVelFromGrid(ipos, airux, airuy, airuz);

		vel[idx] += gvel;
	}
}


__global__ void compsurfacetension_k(farray sf, charray mark, farray phigrax, farray phigray, farray phigraz, float sigma)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		if (mark[idx] != TYPEBOUNDARY)
		{
			int i, j, k;
			getijk(i, j, k, idx);

			float len, h = dparam.cellsize.x;
			float res, grax1, gray1, graz1, grax0, gray0, graz0;
			float3 phigracenter = make_float3(phigrax[idx], phigray[idx], phigraz[idx]);
			len = length(phigracenter);
			if (len == 0)
				res = 0;
			else
			{
				phigracenter /= len;

				if (verifycellidx(i + 1, j, k))
				{
					len = length(make_float3(phigrax(i + 1, j, k), phigray(i + 1, j, k), phigraz(i + 1, j, k)));
					if (len == 0)
						grax1 = phigracenter.x;
					else
						grax1 = phigrax(i + 1, j, k) / len;
				}
				else
					grax1 = phigracenter.x;

				if (verifycellidx(i - 1, j, k))
				{
					len = length(make_float3(phigrax(i - 1, j, k), phigray(i - 1, j, k), phigraz(i - 1, j, k)));
					if (len == 0)
						grax0 = phigracenter.x;
					else
						grax0 = phigrax(i - 1, j, k) / len;
				}
				else
					grax0 = phigracenter.x;

				if (verifycellidx(i, j + 1, k))
				{
					len = length(make_float3(phigrax(i, j + 1, k), phigray(i, j + 1, k), phigraz(i, j + 1, k)));
					if (len == 0)
						gray1 = phigracenter.y;
					else
						gray1 = phigray(i, j + 1, k) / len;
				}
				else
					gray1 = phigracenter.y;

				if (verifycellidx(i, j - 1, k))
				{
					len = length(make_float3(phigrax(i, j - 1, k), phigray(i, j - 1, k), phigraz(i, j - 1, k)));
					if (len == 0)
						gray0 = phigracenter.y;
					else
						gray0 = phigray(i, j - 1, k) / len;
				}
				else
					gray0 = phigracenter.y;

				if (verifycellidx(i, j, k + 1))
				{
					len = length(make_float3(phigrax(i, j, k + 1), phigray(i, j, k + 1), phigraz(i, j, k + 1)));
					if (len == 0)
						graz1 = phigracenter.z;
					else
						graz1 = phigraz(i, j, k + 1) / len;
				}
				else
					graz1 = phigracenter.z;
				if (verifycellidx(i, j, k - 1))
				{
					len = length(make_float3(phigrax(i, j, k - 1), phigray(i, j, k - 1), phigraz(i, j, k - 1)));
					if (len == 0)
						graz0 = phigracenter.z;
					else
						graz0 = phigraz(i, j, k - 1) / len;
				}
				else
					graz0 = phigracenter.z;

				res = (grax1 - grax0 + gray1 - gray0 + graz1 - graz0) / h * 0.5f;
				//res = (grax1-phigracenter.x + gray1-phigracenter.y + graz1-phigracenter.z) / h ;
			}

			sf[idx] = res*sigma;
		}
		else
			sf[idx] = 0;
	}
}

__global__ void enforcesurfacetension_p(float3* ppos, float3 *pvel, char *pflag, int pnum, farray lsmerge, farray sf, farray phigrax, farray phigray, farray phigraz, charray mark, SCENE scene)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] == TYPESOLID/* || pflag[idx]==TYPEAIRSOLO*/ || pflag[idx] == TYPEFLUID)
			return;
		if( (scene != SCENE_MELTANDBOIL&&scene != SCENE_MELTANDBOIL_HIGHRES && pflag[idx] == TYPEAIRSOLO) || ((scene != SCENE_ALL && pflag[idx] == TYPEAIRSOLO)))
			return;

		//1. compute the cell, and get the ls, get sf.
		float3 ipos = ppos[idx];
		float ilsmerge = getScaleFromFrid(ipos, lsmerge);
		float isf = getScaleFromFrid(ipos, sf);
		float3 dir = getVectorFromGrid(ipos, phigrax, phigray, phigraz);
		float lendir = length(dir);
		if (lendir == 0)
			return;
		float3 f;

		dir /= lendir;
		ilsmerge /= lendir;

		//周围最少一个格子是空气的
		int i, j, k;
		getijkfrompos(i, j, k, ipos);
		int cnt = (mark(i, j, k) == TYPEAIR) ? 1 : 0;
		for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
		if (verifycellidx(i + di, j + dj, k + dk))
		if (mark(i + di, j + dj, k + dk) == TYPEAIR)
			cnt++;
		if (cnt == 0)
			return;

		// if(abs(ls_p)<threshold), enforce a surface tension force, change the velocity.
		if (abs(ilsmerge)<dparam.cellsize.x)
		{
			f = -isf*dir;
			pvel[idx] += f*dparam.dt;
		}
	}
}

//标记levelset里比较大的正数，他们是邻近域内没有粒子的
__global__ void markLS_bigpositive(farray ls, charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<(ls.xn*ls.yn*ls.zn))
	{
		ls[idx] = ls[idx] / dparam.cellsize.x;
		if (ls[idx] >1.99f)
		{
			ls[idx] = 5.0f;
			mark[idx] = TYPEAIR;	//标记为需要sweep的单元，并非真正的标记 
		}
		else
			mark[idx] = TYPEFLUID;
	}
}

__global__ void setLSback_bigpositive(farray ls)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<(ls.xn*ls.yn*ls.zn))
	{
		ls[idx] = ls[idx] * dparam.cellsize.x;
	}
}

__global__ void preparels(farray ls, charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<(ls.xn*ls.yn*ls.zn))
	{
		ls[idx] = -ls[idx] / dparam.cellsize.x;
		if (ls[idx] >0)
		{
			ls[idx] = 5.0f;
			mark[idx] = TYPEAIR;	//标记为需要sweep的单元，并非真正的标记 
		}
		else
			mark[idx] = TYPEFLUID;
	}
}

__global__ void setLSback(farray ls)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<(ls.xn*ls.yn*ls.zn))
	{
		ls[idx] = -ls[idx] * dparam.cellsize.x;
	}
}

__global__ void mergeLSAndMarkGrid(farray lsmerge, charray mark, farray lsfluid, farray lsair)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx< dparam.gnum)
	{
		float h = dparam.cellsize.x;

		if (lsair[idx] >4.99f * h)
		{
			lsmerge[idx] = lsfluid[idx];
			if (lsfluid[idx]>0)
				mark[idx] = TYPEVACUUM;
			else
				mark[idx] = TYPEFLUID;
		}
		else if (lsfluid[idx]>4.99f*h)
		{
			lsmerge[idx] = lsair[idx];
			if (lsair[idx]>0)
				mark[idx] = TYPEVACUUM;
			else
				mark[idx] = TYPEAIR;
		}
		else if (lsair[idx]>0.8f*h && lsfluid[idx]>0.8f*h)
		{
			mark[idx] = TYPEVACUUM;
			lsmerge[idx] = min(lsfluid[idx], lsair[idx]);
		}
		else
		{
			lsmerge[idx] = (lsfluid[idx] - lsair[idx])*0.5f;
			if (lsmerge[idx]>0)
				mark[idx] = TYPEAIR;
			else
				mark[idx] = TYPEFLUID;
		}
		//todo: 对于气体将出到水面的时候，ls还是会有问题
		int i, j, k;
		getijk(i, j, k, idx);
		if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1 || k == 0 || k == NZ - 1)
			mark[idx] = TYPEBOUNDARY, lsmerge[idx] = -0.5f*h;
		//todo: debug: 
		//lsmerge[idx] = -lsmerge[idx];
	}
}

__global__ void sweepu_k_bubble(farray outux, farray outuy, farray outuz, farray ux, farray uy, farray uz, farray ls, charray mark, char sweepflag)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	int i, j, k;
	float wx, wy, wz, wsum;		//三个方向上的权重
	if (idx < dparam.gvnum.x)
	{
		//copy
		outux[idx] = ux[idx];

		//ux
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		if (i>1 && i<NX - 1 /*&& j>0 && j<N-1 && k>0 && k<N-1*/)
		{
			if ((mark(i, j, k) != sweepflag && mark(i - 1, j, k) != sweepflag))
			for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
			{
				if (j + dj<0 || j + dj>NY - 1 || k + dk<0 || k + dk >NZ -1)
					continue;
				wx = -di*(ls(i, j, k) - ls(i - 1, j, k));
				if (wx<0)
					continue;
				wy = (ls(i, j, k) + ls(i - 1, j, k) - ls(i, j + dj, k) - ls(i - 1, j + dj, k))*0.5f;
				if (wy<0)
					continue;
				wz = (ls(i, j, k) + ls(i - 1, j, k) - ls(i, j, k + dk) - ls(i - 1, j, k + dk))*0.5f;
				if (wz<0)
					continue;
				wsum = wx + wy + wz;
				if (wsum == 0)
					wx = wy = wz = 1.0f / 3;
				else
					wx /= wsum, wy /= wsum, wz /= wsum;
				outux(i, j, k) = wx*ux(i + di, j, k) + wy* ux(i, j + dj, k) + wz* ux(i, j, k + dk);
			}
		}
	}
	if (idx < dparam.gvnum.y)
	{
		//copy
		outuy[idx] = uy[idx];

		//uy
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		if ( /*i>0 && i<N-1 &&*/ j>1 && j<NY - 1 /*&& k>0 && k<N-1*/)
		{
			if ((mark(i, j, k) != sweepflag && mark(i, j - 1, k) != sweepflag))
			for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
			{
				if (i + di<0 || i + di>NX - 1 || k + dk<0 || k + dk >NZ - 1)
					continue;
				wy = -dj*(ls(i, j, k) - ls(i, j - 1, k));
				if (wy<0)
					continue;
				wx = (ls(i, j, k) + ls(i, j - 1, k) - ls(i + di, j, k) - ls(i + di, j - 1, k))*0.5f;
				if (wx<0)
					continue;
				wz = (ls(i, j, k) + ls(i, j - 1, k) - ls(i, j, k + dk) - ls(i, j - 1, k + dk))*0.5f;
				if (wz<0)
					continue;
				wsum = wx + wy + wz;
				if (wsum == 0)
					wx = wy = wz = 1.0f / 3;
				else
					wx /= wsum, wy /= wsum, wz /= wsum;
				outuy(i, j, k) = wx*uy(i + di, j, k) + wy* uy(i, j + dj, k) + wz* uy(i, j, k + dk);
			}
		}
	}
	if (idx < dparam.gvnum.z)
	{
		//copy
		outuz[idx] = uz[idx];

		//uz
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		if ( /*i>0 && i<N-1 && j>0 && j<N-1 &&*/ k>1 && k<NZ - 1)
		{
			if ((mark(i, j, k) != sweepflag && mark(i, j, k - 1) != sweepflag))
			for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
			{
				if (i + di<0 || i + di >NX - 1 || j + dj<0 || j + dj>NY - 1)
					continue;
				wz = -dk*(ls(i, j, k) - ls(i, j, k - 1));
				if (wz<0)
					continue;
				wy = (ls(i, j, k) + ls(i, j, k - 1) - ls(i, j + dj, k) - ls(i, j + dj, k - 1))*0.5f;
				if (wy<0)
					continue;
				wx = (ls(i, j, k) + ls(i, j, k - 1) - ls(i + di, j, k) - ls(i + di, j, k - 1))*0.5f;
				if (wx<0)
					continue;
				wsum = wx + wy + wz;
				if (wsum == 0)
					wx = wy = wz = 1.0f / 3;
				else
					wx /= wsum, wy /= wsum, wz /= wsum;
				outuz(i, j, k) = wx*uz(i + di, j, k) + wy* uz(i, j + dj, k) + wz* uz(i, j, k + dk);
			}
		}
	}
}


//修正粒子的位置，当气体粒子跑到流体中时，"拉"它回来，反之亦然
__global__ void correctbubblepos(farray ls, farray phigrax, farray phigray, farray phigraz, float3 *ppos, char* pflag, int pnum, float *pphi)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		char iflag = pflag[idx];
		//test. todo. debug
		if (iflag == TYPEAIRSOLO || iflag == TYPESOLID)
			return;

		float3 ipos = ppos[idx];
		int s = (iflag == TYPEFLUID) ? -1 : 1;
		float d, dirlen, rs = 0.5f*dparam.cellsize.x;
		float3 dir = getVectorFromGrid(ipos, phigrax, phigray, phigraz);
		dirlen = length(dir);
		if (dirlen == 0)
			return;
		else
			dir = normalize(dir);
		d = getScaleFromFrid(ipos, ls) / dirlen;
		//test
		// 		if( s*d<0 )
		// 			ipos=ipos +rs*dir;
		//debug.
		pphi[idx] = d;

		//todo: 这里有问题
		if (s*d<0 && abs(d)<0.5f*dparam.cellsize.x)	//wrong way
		{
			if (iflag == TYPEAIR&& abs(d)>0.3f*dparam.cellsize.x)	//气体粒子只在错位比较明显的情况下才纠正，主要是为了防止气泡体积的收缩。
				ipos = ipos - d*dir;
			else if (iflag == TYPEFLUID)
			{
				ipos = ipos - d*dir;

				dir = getVectorFromGrid(ipos, phigrax, phigray, phigraz);
				dirlen = length(dir);
				if (dirlen == 0)
					return;
				else
					dir = normalize(dir);
				d = getScaleFromFrid(ipos, ls) / dirlen;

				ipos = ipos + s*(rs - s*d)*dir;
			}
			//	cnt++;
		}
		else if (iflag == TYPEFLUID && s*d<rs*0.5f && s*d >= 0)		//todo: rs*0.5f有点小问题，但不加这个0.5的话流体的体积会变化明显
		{
			ipos = ipos + s*(rs - s*d)*dir;
		}
		ppos[idx] = ipos;
	}
}

//修正粒子的位置，当气体粒子跑到流体中时，"拉"它回来，反之亦然.
//这里修正液体粒子位置时用的是气体的ls
__global__ void correctbubblepos_air(farray lsmerge, farray phigrax, farray phigray, farray phigraz, farray lsair, farray phigrax_air, farray phigray_air, farray phigraz_air, float3 *ppos, char* pflag, int pnum, float *pphi)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		char iflag = pflag[idx];
		//test. todo. debug
		if (iflag == TYPEAIRSOLO || iflag == TYPESOLID)
			return;

		float3 ipos = ppos[idx];
		int s = (iflag == TYPEFLUID) ? -1 : 1;
		float d, dirlen, rs = 0.5f*dparam.cellsize.x;
		float3 dir = getVectorFromGrid(ipos, phigrax, phigray, phigraz);
		dirlen = length(dir);
		if (dirlen == 0)
			return;
		else
			dir = normalize(dir);
		d = getScaleFromFrid(ipos, lsmerge) / dirlen;
		//test
		// 		if( s*d<0 )
		// 			ipos=ipos +rs*dir;
		//debug.
		pphi[idx] = d;

		//todo: 这里有问题
		if (s*d<0 && abs(d)<0.5f*dparam.cellsize.x)	//wrong way
		{
			if (iflag == TYPEAIR&& abs(d)>0.3f*dparam.cellsize.x)	//气体粒子只在错位比较明显的情况下才纠正，主要是为了防止气泡体积的收缩。
				ipos = ipos - d*dir;

			//	cnt++;
		}
		if (iflag == TYPEFLUID)	//对液体粒子使用气体的level set来处理，慢慢把液体“挤出”气泡之外，使得lsmerge计算更为准确
		{
			dir = getVectorFromGrid(ipos, phigrax_air, phigray_air, phigraz_air);
			dirlen = length(dir);
			if (dirlen == 0)
				return;
			else
				dir = normalize(dir);
			d = getScaleFromFrid(ipos, lsair) / dirlen;

			if (d<-1.3f*rs)
				ipos = ipos - (d - rs)*dir;
		}

		ppos[idx] = ipos;
	}
}

//根据levelset计算梯度场，相当于一个方向
__global__ void computePhigra(farray phigrax, farray phigray, farray phigraz, farray ls)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		float h = dparam.cellsize.x;
		float lsx1, lsx0, lsy1, lsy0, lsz1, lsz0, lscenter = ls[idx];
		lsx1 = (verifycellidx(i + 1, j, k)) ? ls(i + 1, j, k) : lscenter;
		lsx0 = (verifycellidx(i - 1, j, k)) ? ls(i - 1, j, k) : lscenter;
		lsy1 = (verifycellidx(i, j + 1, k)) ? ls(i, j + 1, k) : lscenter;
		lsy0 = (verifycellidx(i, j - 1, k)) ? ls(i, j - 1, k) : lscenter;
		lsz1 = (verifycellidx(i, j, k + 1)) ? ls(i, j, k + 1) : lscenter;
		lsz0 = (verifycellidx(i, j, k - 1)) ? ls(i, j, k - 1) : lscenter;

		//todo: 这里需要考虑一下
		phigrax[idx] = ((lsx1 - lsx0)*0.5f) / h;
		phigray[idx] = ((lsy1 - lsy0)*0.5f) / h;
		phigraz[idx] = ((lsz1 - lsz0)*0.5f) / h;

		//phigrax[idx] = (lsx1-lscenter)/h;
		//phigray[idx] = (lsy1-lscenter)/h;
		//phigraz[idx] = (lsz1-lscenter)/h;
	}
}

__global__ void copyParticle2GL_phi(float3* ppos, char *pflag, float *pmass, float *pTemperature, int pnum, float *renderpos, float *rendercolor,
	farray ls, farray phigrax, farray phigray, farray phigraz, char typeflag, float Tmax, float Tmin)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//todo:
		if (pflag[idx] == typeflag/* || ppos[idx].y<NY*0.5f*dparam.cellsize.x */)
		{
			renderpos[idx * 3] = -2.0f;
			renderpos[idx * 3 + 1] = 0.0f;
			renderpos[idx * 3 + 2] = 0.0f;

			float3 color = make_float3(0.0f);
			rendercolor[idx * 3] = color.x;
			rendercolor[idx * 3 + 1] = color.y;
			rendercolor[idx * 3 + 2] = color.z;
			return;
		}
		renderpos[idx * 3] = ppos[idx].x;
		renderpos[idx * 3 + 1] = ppos[idx].y;
		renderpos[idx * 3 + 2] = ppos[idx].z;

		float3 color;

		if (pflag[idx] == TYPEAIR)
			color = mapColorBlue2Red(0.0f);
		else if (pflag[idx] == TYPEFLUID)
			color = mapColorBlue2Red(2.0f);
		else	 if (pflag[idx] == TYPESOLID)
			color = mapColorBlue2Red(4.0f);
		else
			color = mapColorBlue2Red(6.0f);
		//color=mapColorBlue2Red( (pTemperature[idx]-Tmin)/(Tmax-Tmin)*6.0f );


		rendercolor[idx * 3] = color.x;
		rendercolor[idx * 3 + 1] = color.y;
		rendercolor[idx * 3 + 2] = color.z;
	}
}

//压强与速度的计算，加入surface tension. [2005]Discontinuous Fluids
__global__ void subGradPress_bubble(farray p, farray ux, farray uy, farray uz, farray sf, farray lsmerge, charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int i, j, k;
	float h = dparam.cellsize.x;
	float J = 0.0f, theta;
	if (idx<dparam.gvnum.x)
	{
		J = 0.0f;
		//ux
		getijk(i, j, k, idx, NX + 1, NY, NZ);
		if (i>0 && i<NX)		//look out for this condition
		{
			if ((mark(i, j, k) == TYPEAIR && mark(i - 1, j, k) == TYPEFLUID) || (mark(i, j, k) == TYPEFLUID && mark(i - 1, j, k) == TYPEAIR))
			{
				theta = (0.0f - lsmerge(i - 1, j, k)) / (lsmerge(i, j, k) - lsmerge(i - 1, j, k));
				J = theta*sf(i - 1, j, k) + (1.0f - theta)*sf(i, j, k);
			}
			ux(i, j, k) -= (p(i, j, k) - p(i - 1, j, k) - J) / h;
		}
	}
	if (idx<dparam.gvnum.y)
	{
		J = 0.0f;
		//uy
		getijk(i, j, k, idx, NX, NY + 1, NZ);
		if (j>0 && j<NY)		//look out for this condition
		{
			if ((mark(i, j, k) == TYPEAIR && mark(i, j - 1, k) == TYPEFLUID) || (mark(i, j, k) == TYPEFLUID && mark(i, j - 1, k) == TYPEAIR))
			{
				theta = (0.0f - lsmerge(i, j - 1, k)) / (lsmerge(i, j, k) - lsmerge(i, j - 1, k));
				J = theta*sf(i, j - 1, k) + (1.0f - theta)*sf(i, j, k);
			}
			uy(i, j, k) -= (p(i, j, k) - p(i, j - 1, k) - J) / h;
		}
	}
	if (idx<dparam.gvnum.z)
	{
		J = 0.0f;
		//uz
		getijk(i, j, k, idx, NX, NY, NZ + 1);
		if (k>0 && k<NZ)		//look out for this condition
		{
			if ((mark(i, j, k) == TYPEAIR && mark(i, j, k - 1) == TYPEFLUID) || (mark(i, j, k) == TYPEFLUID && mark(i, j, k - 1) == TYPEAIR))
			{
				theta = (0.0f - lsmerge(i, j, k - 1)) / (lsmerge(i, j, k) - lsmerge(i, j, k - 1));
				J = theta*sf(i, j, k - 1) + (1.0f - theta)*sf(i, j, k);
			}
			uz(i, j, k) -= (p(i, j, k) - p(i, j, k - 1) - J) / h;
		}
	}
}

__global__ void sweepVacuum(charray mark)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		if (mark[idx] != TYPEAIR)
			return;
		//mark
		for (int di = -1; di <= 1; di += 2) for (int dj = -1; dj <= 1; dj += 2) for (int dk = -1; dk <= 1; dk += 2)
		if (mark(i + di, j + dj, k + dk) == TYPEVACUUM)
			mark[idx] = TYPEVACUUM;

	}
}

__global__ void markDeleteAirParticle(float3* ppos, char* pflag, float *pmass, uint *preservemark, int pnum, charray mark, farray lsmerge, farray lsair, uint *cnt)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		//fluid and solid particles are preserved, air and airsolo particles are verified.
		if (pflag[idx] == TYPESOLID)
		{
			preservemark[idx] = 1;
			return;
		}

		int i, j, k;
		getijkfrompos(i, j, k, ppos[idx]);

		if (pflag[idx] == TYPEFLUID)
		{
			float lsm = getScaleFromFrid(ppos[idx], lsmerge);
			float lsa = getScaleFromFrid(ppos[idx], lsair);
			if ( /*lsm>1.2f*dparam.cellsize.x || */lsa<-1.0*dparam.cellsize.x)
				preservemark[idx] = 0, cnt[0]++;
			else
				preservemark[idx] = 1;
			return;
		}

		int cnt = 0;
		for (int di = -1; di <= 1; di += 1) for (int dj = -1; dj <= 1; dj += 1) for (int dk = -1; dk <= 1; dk += 1)
		if (verifycellidx(i + di, j + dj, k + dk) && mark(i + di, j + dj, k + dk) == TYPEVACUUM)
			cnt++;
		if (cnt == 0 && pmass[idx]>0.000001f)		//notice: 这里附带的删除了质量过小的气体粒子，与气体粒子的被吸收有关
			preservemark[idx] = 1;
		else
			preservemark[idx] = 0;
	}
}

// compact voxel array
__global__ void deleteparticles(uint *preserveflag, uint *preserveflagscan, int pnum, float3 *outpos, float3 *pos,
	float3 *outvel, float3 *vel, float *outmass, float* mass, char *outflag, char *flag, float *outTemperature, float *temperature, float *outheat, float *heat,
	float *outsolubility, float *solubility, float *outgascontain, float *gascontain)
{
	uint idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;

	if (idx<pnum)
	{
		if (preserveflag[idx] == 1)
		{
			//deleteflagscan 存的是删除某些粒子之后的"索引".
			uint outidx = preserveflagscan[idx];
			outpos[outidx] = pos[idx];
			outvel[outidx] = vel[idx];
			outmass[outidx] = mass[idx];
			outflag[outidx] = flag[idx];
			outTemperature[outidx] = temperature[idx];
			outheat[outidx] = heat[idx];
			outsolubility[outidx] = solubility[idx];
			outgascontain[outidx] = gascontain[idx];
		}
	}
}

__device__ int cntairparticle(float3 *ppos, char *pflag, int igrid, uint *gridstart, uint *gridend, const float3 &ipos, float r)
{
	uint start = gridstart[igrid];
	int res = 0;
	float dis;
	if (start == CELL_UNDEF)
		return res;
	for (int p = start; p<gridend[igrid]; p++)
	{
		dis = length(ppos[p] - ipos);
		if (dis<r && (pflag[p] == TYPEAIR || pflag[p] == TYPEAIRSOLO))
		{
			++res;
		}
	}
	return res;
}

__device__ inline bool isInBoundaryCell(int x, int y, int z)
{
	int level = 2;
	if (x <= level || x >= NX - 1 - level || y <= level || y >= NY - 1 - level)
		return true;
	else
		return false;
}

__global__ void verifySoloAirParticle(float3 *ppos, float3 *pvel, char *pflag, int pnum, farray lsmerge, farray airux, farray airuy, farray airuz, uint *gridstart, uint *gridend, SCENE scene)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		char iflag = pflag[idx];
		if (iflag == TYPEFLUID || iflag == TYPESOLID)	//TYPEAIR, TYPEAIRSOLO can go on.
			return;

		float3 ipos = ppos[idx];
		float ls = getScaleFromFrid(ipos, lsmerge);
		float h = dparam.cellsize.x;
		int i, j, k;
		getijkfrompos(i, j, k, ipos);

		//a key adjustment, the tolerent will affect the result directly.
		int cnt = 0;
		for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			cnt += cntairparticle(ppos, pflag, getidx(i + di, j + dj, k + dk), gridstart, gridend, ipos, h);

		float tol1 = -1.45f, tol2 = -0.5f;
		if (scene == SCENE_MELTANDBOIL || scene == SCENE_MELTANDBOIL_HIGHRES || scene==SCENE_ALL)
			tol1 = 0.05f, tol2 = -0.8f;
		else if (scene == SCENE_INTERACTION)
			tol1 = 0.2f, tol2 = -0.5f;

		if ((cnt >= 10 || ls>tol1*h) && pflag[idx] == TYPEAIRSOLO && !isInBoundaryCell(i, j, k))		//decide whether the air solo particle should  be transfered to air particle.
		{
			if (cnt >= 3)
				pflag[idx] = TYPEAIR;
		}
		else if (iflag == TYPEAIR && (isInBoundaryCell(i, j, k) || ls<tol2*h || cnt <= 1))
		{
			//todo: 插值速度 or not???
			//pvel[idx]= pvel[idx]*0.8f + 0.2f*getParticleVelFromGrid(ipos,airux,airuy,airuz);
			pvel[idx] = getParticleVelFromGrid(ipos, airux, airuy, airuz);
			pflag[idx] = TYPEAIRSOLO;
		}
	}
}

__device__ float sumdensity(float3 ipos, float h2, int grididx, float3 *ppos, char *pflag, uint *gridstart, uint *gridend)
{
	float res = 0;
	uint start = gridstart[grididx];
	if (start == CELL_UNDEF)
		return res;
	float dist2;
	for (uint p = start; p<gridend[grididx]; p++)
	{
		// notice: should include liquid particle, not just spray particle.
		if (pflag[p] != TYPEAIR && pflag[p] != TYPEAIRSOLO)
			continue;
		dist2 = dot(ppos[p] - ipos, ppos[p] - ipos);
		if (dist2<h2)
			res += pow(h2 - dist2, 3.0f);	//todo: m0 or pmass[p]?
	}
	return res;
}

__global__ void calcDensPress_Air(float3* ppos, float *pdens, float *ppress, char* pflag, int pnum, uint *gridstart, uint *gridend)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] != TYPEAIR && pflag[idx] != TYPEAIRSOLO)
			return;

		float3 ipos = ppos[idx];
		float h = dparam.cellsize.x;		//todo: set support radius, key part.
		float h2 = h*h;
		int i, j, k;
		getijkfrompos(i, j, k, ipos);

		float dens = 0;
		for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			dens += sumdensity(ipos, h2, getidx(i + di, j + dj, k + dk), ppos, pflag, gridstart, gridend);

		dens *= dparam.airm0 * dparam.poly6kern;
		if (dens == 0) dens = 1.0f;
		pdens[idx] = 1.0f / dens;
		ppress[idx] = 1.5f * (dens - dparam.waterrho*0.5f);
	}
}


__device__ float3 sumforce(float3 *ppos, float3 *pvel, float *ppress, float *pdens, char *pflag, int grididx, uint *gridstart, uint *gridend,
	float3 ipos, float3 ivel, float ipress, float idens, float h, float kvis)
{
	uint start = gridstart[grididx];
	float3 res = make_float3(0.0f), dir;
	float dis, c, pterm, dterm;// kattrct=0.0f, 
	if (start == CELL_UNDEF)
		return res;
	float vterm = dparam.lapkern * kvis;

	for (uint p = start; p<gridend[grididx]; p++)
	{
		dir = ipos - ppos[p];
		dis = length(dir);
		if (dis>0 && dis<h && (pflag[p] == TYPEAIRSOLO || pflag[p] == TYPEAIR))
		{
			c = h - dis;
			pterm = -0.5f * c * dparam.spikykern * (ipress + ppress[p]) / dis;
			dterm = c * idens * pdens[p];
			res += (pterm * dir + vterm * (pvel[p] - ivel)) * dterm;
		}
	}
	return res;
}
__global__ void enforceForceSoloAirP(float3 *ppos, float3 *pvel, float *pdens, float *ppress, char *pflag, int pnum, uint *gridstart, uint *gridend, float viscositySPH, float maxVelForBubble)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] != TYPEAIRSOLO && pflag[idx] != TYPEAIR)
			return;

		float3 ipos = ppos[idx];
		float3 ivel = pvel[idx];
		float ipress = ppress[idx], idens = pdens[idx];
		float h = dparam.cellsize.x;
		//float kvis=0.0f;	

		int i, j, k;
		float3 force = make_float3(0.0f);
		getijkfrompos(i, j, k, ipos);

		int width = 1;
		for (int di = -width; di <= width; di++) for (int dj = -width; dj <= width; dj++) for (int dk = -width; dk <= width; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			force += sumforce(ppos, pvel, ppress, pdens, pflag, getidx(i + di, j + dj, k + dk), gridstart, gridend, ipos, ivel, ipress, idens, h, viscositySPH);

		//todo: 直接更新速度和位置??
		force *= dparam.airm0;
		//force = make_float3(0);

		ivel += force*dparam.dt;
		ipos += ivel*dparam.dt;

		//restrict the vel below a threshold.
		// 		if( length(ivel) > maxVelForBubble )
		// 			ivel = normalize(ivel) * maxVelForBubble;
		// 	
		//	advect particle, using rho!!!!
		//	ppos[idx]=ipos;
		pvel[idx] = ivel;
	}
}


__device__ float sumdensity_SLCouple(float3 ipos, float h2, int grididx, float3 *ppos, char *pflag, uint *gridstart, uint *gridend)
{
	float res = 0;
	uint start = gridstart[grididx];
	if (start == CELL_UNDEF)
		return res;
	float dist2;
	for (uint p = start; p<gridend[grididx]; p++)
	{
		dist2 = dot(ppos[p] - ipos, ppos[p] - ipos);
		if (dist2<h2)
			res += pow(h2 - dist2, 3.0f);
	}
	return res;
}

//solid-liquid coupling, in SPH framework
__global__ void calcDensPressSPH_SLCouple(float3* ppos, float *pdens, float *ppress, char* pflag, int pnum, uint *gridstart, uint *gridend)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		float3 ipos = ppos[idx];
		float h = dparam.cellsize.x;		//todo: set support radius, key part.
		float h2 = h*h;
		int i, j, k;
		getijkfrompos(i, j, k, ipos);

		float dens = 0;
		for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			dens += sumdensity_SLCouple(ipos, h2, getidx(i + di, j + dj, k + dk), ppos, pflag, gridstart, gridend);

		dens *= dparam.m0 * dparam.poly6kern;
		if (dens == 0) dens = 1.0f;
		pdens[idx] = 1.0f / dens;
		ppress[idx] = 1.5f * (dens - dparam.waterrho);
	}
}

__device__ float3 sumforce_SLCouple(float3 *ppos, float3 *pvel, float *ppress, float *pdens, char *pflag, int grididx, uint *gridstart, uint *gridend,
	float3 ipos, float3 ivel, float ipress, float idens, float h, float kvis)
{
	uint start = gridstart[grididx];
	float3 res = make_float3(0.0f), dir;
	float dis, c, pterm, dterm;// kattrct=0.0f, kvis=0.0f;
	if (start == CELL_UNDEF)
		return res;
	float vterm = dparam.lapkern * kvis;

	for (uint p = start; p<gridend[grididx]; p++)
	{
		dir = ipos - ppos[p];
		dis = length(dir);
		if (dis>0 && dis<h)
		{
			c = h - dis;
			pterm = -0.5f * c * dparam.spikykern * (ipress + ppress[p]) / dis;
			dterm = c * idens * pdens[p];
			res += (pterm * dir + vterm * (pvel[p] - ivel)) * dterm;
		}
	}
	return res;
}
__global__ void enforceForceSPH_SLCouple(float3 *ppos, float3 *pvel, float *pdens, float *ppress, char *pflag, int pnum, uint *gridstart, uint *gridend, float viscositySPH)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] != TYPEFLUID)	//只有fluid计算，solid不在这里更新
			return;

		float3 ipos = ppos[idx];
		float3 ivel = pvel[idx];
		float ipress = ppress[idx], idens = pdens[idx];
		float h = dparam.cellsize.x;
		//float kvis=0.0f;	

		int i, j, k;
		float3 force = make_float3(0.0f);
		getijkfrompos(i, j, k, ipos);

		int width = 1;
		for (int di = -width; di <= width; di++) for (int dj = -width; dj <= width; dj++) for (int dk = -width; dk <= width; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			force += sumforce_SLCouple(ppos, pvel, ppress, pdens, pflag, getidx(i + di, j + dj, k + dk), gridstart, gridend, ipos, ivel, ipress, idens, h, viscositySPH);

		//	force=make_float3(0.0f);
		//todo: 直接更新速度和位置??
		//add gravity here? or external force part;
		force *= dparam.m0;
		//force = make_float3(0);

		ivel += force*dparam.dt;
		ipos += ivel*dparam.dt;

		//	advect particle, using rho!!!!
		ppos[idx] = ipos;
		pvel[idx] = ivel;
	}
}

__global__ void updateFixedHeat(farray fixedHeat, int frame)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		if (i >= NX / 4 && i<NX*0.75 && j >= NY / 4 && j<NY*0.75 && k <= 3 /*k<=20 && k>=19*/)
			fixedHeat[idx] = 273.0f + 100.0f * min(frame / 40.f, 1.0f);
		else
			fixedHeat[idx] = UNDEF_TEMPERATURE;
	}
}

__global__ void addHeatAtBottom(farray Tp, int frame, float heatIncreaseBottom)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		if (i >= 1 && i<NX - 1 && j >= 1 && j<NY - 1 && k <= 3 /*k<=20 && k>=19*/)
			Tp[idx] += heatIncreaseBottom;//1.5f;
		//Tp[idx] = 350.0f;//273.0f + 100.0f * min(frame/40.f, 1.0f );
		Tp[idx] = min(378.0f, Tp[idx]);
	}
}

//
__global__ void compb_heat(farray Tp_old, farray Tp, farray fixedheat, charray mark, float *heatAlphaArray)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx <dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);

		float alpha = heatAlphaArray[mark[idx]];

		//如果有固定的温度，那么tp与b都要根据这个fixedheat来计算
		// 		if( fixedheat[idx]!=UNDEF_TEMPERATURE )
		// 			Tp[idx]=fixedheat[idx], Tp_old[idx] = fixedheat[idx]*dparam.cellsize.x*dparam.cellsize.x/alpha/dparam.dt;
		// 		else
		Tp_old[idx] = Tp[idx] * dparam.cellsize.x*dparam.cellsize.x / alpha / dparam.dt;
	}
}

//z = Ax: A is a sparse matrix, representing the left hand item of Poisson equation.
__global__ void computeAx_heat(farray ans, charray mark, farray x, int n, float *heatAlphaArray, farray fixedHeat, SCENE scene)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<n)
	{
		float h = dparam.cellsize.x;
		float dt = dparam.dt;

		float alpha = heatAlphaArray[mark[idx]];

		if (mark[idx] != TYPEBOUNDARY/* && mark[idx]!=TYPEVACUUM*/)
		{
			int i, j, k;
			getijk(i, j, k, idx);

			float center = x[idx];
			float sum = (h*h / alpha / dt + 6.0f)*center;

			//trick: 决定要不要让freeair参与计算
			if (scene == SCENE_BOILING || scene == SCENE_BOILING_HIGHRES || scene == SCENE_MELTANDBOIL || scene == SCENE_MELTANDBOIL_HIGHRES || scene ==SCENE_ALL)
			{
				sum -= ((mark(i + 1, j, k) == TYPEBOUNDARY || mark(i + 1, j, k) == TYPEVACUUM) ? center : x(i + 1, j, k));
				sum -= ((mark(i, j + 1, k) == TYPEBOUNDARY || mark(i, j + 1, k) == TYPEVACUUM) ? center : x(i, j + 1, k));
				sum -= ((mark(i, j, k + 1) == TYPEBOUNDARY || mark(i, j, k + 1) == TYPEVACUUM) ? center : x(i, j, k + 1));
				sum -= ((mark(i - 1, j, k) == TYPEBOUNDARY || mark(i - 1, j, k) == TYPEVACUUM) ? center : x(i - 1, j, k));
				sum -= ((mark(i, j - 1, k) == TYPEBOUNDARY || mark(i, j - 1, k) == TYPEVACUUM) ? center : x(i, j - 1, k));
				sum -= ((mark(i, j, k - 1) == TYPEBOUNDARY || mark(i, j, k - 1) == TYPEVACUUM) ? center : x(i, j, k - 1));
			}
			else
			{
				sum -= ((mark(i + 1, j, k) == TYPEBOUNDARY) ? center : x(i + 1, j, k));
				sum -= ((mark(i, j + 1, k) == TYPEBOUNDARY) ? center : x(i, j + 1, k));
				sum -= ((mark(i, j, k + 1) == TYPEBOUNDARY) ? center : x(i, j, k + 1));
				sum -= ((mark(i - 1, j, k) == TYPEBOUNDARY) ? center : x(i - 1, j, k));
				sum -= ((mark(i, j - 1, k) == TYPEBOUNDARY) ? center : x(i, j - 1, k));
				sum -= ((mark(i, j, k - 1) == TYPEBOUNDARY) ? center : x(i, j, k - 1));
			}

			ans[idx] = sum;
		}
	}
}


//Ans = x + a*y
__global__ void pcg_op_heat(charray A, farray ans, farray x, farray y, float a, int n)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<n)
	{
		//	if( A[idx]==TYPEFLUID || A[idx]==TYPEAIR )
		if (A[idx] != TYPEBOUNDARY)
			ans[idx] = x[idx] + a*y[idx];
		else
			ans[idx] = 0.0f;
	}
}


__global__ void setBoundaryHeat(farray tp)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		if (i == NX - 1)
			tp[idx] = tp(i - 1, j, k);
		else if (i == 0)
			tp[idx] = tp(i + 1, j, k);
		else if (j == NY - 1)
			tp[idx] = tp(i, j - 1, k);
		else if (j == 0)
			tp[idx] = tp(i, j + 1, k);
		else if (k == NZ - 1)
			tp[idx] = tp(i, j, k - 1);
		else if (k == 0)
			tp[idx] = tp(i, j, k + 1);
	}
}

__global__ void compTpChange(farray tp, farray tpsave, charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		if (mark[idx] != TYPEBOUNDARY)
			tpsave[idx] = tp[idx] - tpsave[idx];
		else
			tpsave[idx] = 0;
	}
}

__device__ void sumHeat(float &heatsum, float &weight, float3 gpos, float3 *pos, float *pTemperature,
	uint *gridstart, uint  *gridend, int gidx)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis2, w, RE = 1.4;
	float scale = 1 / dparam.cellsize.x;
	for (uint p = start; p<end; ++p)
	{
		dis2 = dot(pos[p] * scale - gpos, pos[p] * scale - gpos);		//scale is necessary.
		w = sharp_kernel(dis2, RE);
		weight += w;
		heatsum += w*pTemperature[p];
	}
}

__global__ void mapHeatp2g_hash(float3 *ppos, float *pTemperature, int pnum, farray heat, uint* gridstart, uint *gridend, float defaulttemperature)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		float weight = 0.0f, heatsum = 0;
		float3 gpos;
		getijk(i, j, k, idx);
		gpos.x = i + 0.5, gpos.y = j + 0.5, gpos.z = k + 0.5;
		for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
			sumHeat(heatsum, weight, gpos, ppos, pTemperature, gridstart, gridend, getidx(i + di, j + dj, k + dk));
		heatsum = (weight>0) ? (heatsum / weight) : defaulttemperature;
		heat(i, j, k) = heatsum;
	}
}

__global__ void mapHeatg2p(float3 *ppos, char *parflag, float *pTemperature, int pnum, farray Tchange, farray T, float defaultSolidT, float alphaTempTrans)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//pos-->grid xyz
		float3 ipos = ppos[idx];
		pTemperature[idx] = alphaTempTrans*(pTemperature[idx] + getScaleFromFrid(ipos, Tchange)) + (1 - alphaTempTrans)*getScaleFromFrid(ipos, T);		//use a scheme like FLIP, update the particle temperature by heat change.
	}
}

__global__ void mapHeatg2p_MeltAndBoil(float3 *ppos, char *parflag, float *pTemperature, int pnum, farray Tchange, farray T, float defaultSolidT, float alphaTempTrans)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		//pos-->grid xyz
		float3 ipos = ppos[idx];
		float newtemp = alphaTempTrans*(pTemperature[idx] + getScaleFromFrid(ipos, Tchange)) + (1 - alphaTempTrans)*getScaleFromFrid(ipos, T);		//use a scheme like FLIP, update the particle temperature by heat change.
		if (parflag[idx] == TYPESOLID)
			pTemperature[idx] = 0.95f*(pTemperature[idx]) + 0.05f*newtemp;
		else
			pTemperature[idx] = newtemp;
	}
}

__global__ void initHeatParticle(float *pTemperature, float *pHeat, float defaultSolidT, float defaultLiquidT, float LiquidHeatTh, char *pflag, int pnum)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] == TYPESOLID)
		{
			pTemperature[idx] = defaultSolidT;
			pHeat[idx] = 0;
		}
		else
		{
			pTemperature[idx] = defaultLiquidT;
			pHeat[idx] = LiquidHeatTh;
		}
	}
}

//Temperature0=273.15K, Solubility0=1.0f (每1个流体粒子里含的气体够生成一个完事的气体粒子)
__global__ void initsolubility_k(float *psolubility, float* pgascontain, float *ptemperature, char *pflag, int pnum, float Solubility0, float Temperature0, float dissolvegasrate, float initgasrate)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] == TYPEFLUID || pflag[idx] == TYPESOLID)
		{
			psolubility[idx] = dissolvegasrate*dparam.airm0 * exp(1018.9f*(1 / ptemperature[idx] - 1 / Temperature0));	//todo: adjust the parameter.
			pgascontain[idx] = initgasrate*psolubility[idx];
		}
		else
		{
			psolubility[idx] = 0;
			pgascontain[idx] = 0;
		}
	}
}

//Temperature0=273.15K, Solubility0=1.0f (每1个流体粒子里含的气体够生成一个完事的气体粒子)
__global__ void updatesolubility(float *psolubility, float *ptemperature, char *pflag, int pnum, float Solubility0, float Temperature0, float dissolvegasrate)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] == TYPEFLUID)
			psolubility[idx] = dissolvegasrate*dparam.airm0 * exp(1018.9f*(1 / ptemperature[idx] - 1 / Temperature0));	//todo: adjust the parameter.
	}
}

//addparnums初始化应该是0
__global__ void GenerateGasParticle_k(float *psolubility, float *paircontain, float3 *ppos, float3 *pvel, float *pmass, char *pflag, float *pTemperature, float *pLHeat,
	int pnum, uint *gridstart, uint *gridend, int *addparnums, float *randfloat, int randcnts, int frame, farray gTemperature, float LiquidHeatTh,
	int *seedcell, int seednum, float vaporGenRate)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		float gcontain = 0, gsolubility = 0, gairexist = 0;
		int liquidParCnt = 0, gasParCnt = 0;
		float airparticlemass0 = dparam.airm0;		//todo
		float vaporsum = 0;//, vaporrate = 0.01f;
		float3 gaspos = make_float3(0), gasvel = make_float3(0);
		int i, j, k;
		getijk(i, j, k, idx);

		if (k <= 1 || isInBoundaryCell(i, j, k))	return;		//最下面的一行不生成气泡粒子

		float3 gpos = make_float3(i, j, k)*dparam.cellsize.x;

		uint start = gridstart[idx];
		if (start == CELL_UNDEF)
			return;
		//1. 统计气体含量、流体粒子含有的气体量、可溶解量
		for (int p = start; p<gridend[idx]; p++)
		{
			if (pflag[p] == TYPEFLUID)
			{
				gcontain += paircontain[p];
				gsolubility += psolubility[p];

				vaporsum += max(0.0f, pLHeat[p] - LiquidHeatTh) * vaporGenRate * airparticlemass0;

				liquidParCnt++;
			}
			else if (pflag[p] == TYPEAIRSOLO || pflag[p] == TYPEAIR)
			{
				gairexist += pmass[p];
				gaspos += ppos[p];
				gasvel += pvel[p];
				gasParCnt++;
			}
		}

		bool hasseed = false;
		for (int i = 0; i<seednum; i++)
		if (seedcell[i] == idx) hasseed = true;

		//如有必要，增加一个气体粒子
		int addcnt = 0;
		int randbase = (idx*frame) % (randcnts - 200);
		//randpos and randfloat are in [0,1]
		float3 randpos = make_float3(randfloat[(randbase + addcnt++) % randcnts], randfloat[(randbase + addcnt++) % randcnts], randfloat[(randbase + addcnt++) % randcnts]);
		float randnum = randfloat[(randbase + addcnt++) % randcnts];
		float r = dparam.cellsize.x * 0.25f;
		if (gcontain - gsolubility + vaporsum > airparticlemass0 && (hasseed || gasParCnt>0))
		{
			int addindex = atomicAdd(&addparnums[0], 1) + pnum;
			pmass[addindex] = airparticlemass0;//dparam.m0;	//todo:
			if (gasParCnt>0)
			{
				ppos[addindex] = gaspos / gasParCnt + (max(0.5f, randnum)*r) * (randpos - make_float3(0.5f)) * 2;	//与凝结核有关
				pvel[addindex] = make_float3(0.0f);//gasvel/gasParCnt;			//与已有的气体粒子有关	
			}
			else
			{
				ppos[addindex] = gpos + dparam.cellsize.x*randpos;
				pvel[addindex] = make_float3(0.0f);
			}
			pflag[addindex] = TYPEAIRSOLO;
			pTemperature[addindex] = gTemperature[idx];		//网格温度
			pLHeat[addindex] = 0;		//气体粒子的heat无所谓
			paircontain[addindex] = 0.0f;
			psolubility[addindex] = 0.0f;

			//重置液体粒子的气体含量
			for (int p = start; p<gridend[idx]; p++)
			{
				if (pflag[p] == TYPEFLUID)
				{
					paircontain[p] = min(paircontain[p], psolubility[p]);
					pLHeat[p] = min(pLHeat[p], LiquidHeatTh);
					//todo: decrease the liquids mass.
				}
			}
		}

	}
}

//addparnums初始化应该是0
__global__ void updatebubblemass(float *psolubility, float *paircontain, float3 *ppos, float *pmass, char *pflag, int pnum, uint *gridstart, uint *gridend)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum / 8)	//每个线程负责8个格子
	{
		float gcontain = 0, gsolubility = 0, gairexist = 0;
		int fpcnt = 0, apcnt = 0;
		float airparticlemass0 = dparam.airm0;		//todo
		int i, j, k;
		getijk(i, j, k, idx, NX / 2, NY / 2, NZ / 2);
		i *= 2, j *= 2, k *= 2;
		//	float3 gpos;
		int gidx;
		for (int di = 0; di <= 1; di++) for (int dj = 0; dj <= 1; dj++) for (int dk = 0; dk <= 1; dk++)
		{
			gidx = getidx(i + di, j + dj, k + dk);
			//	gpos=make_float3(i+di,j+dj,k+dk)*dparam.cellsize.x;
			if (gridstart[gidx] == CELL_UNDEF)	continue;
			//1. 统计气体含量、流体粒子含有的气体量、可溶解量
			for (int p = gridstart[gidx]; p<gridend[gidx]; p++)
			{
				if (pflag[p] == TYPEFLUID)
				{
					gcontain += paircontain[p];
					gsolubility += psolubility[p];

					fpcnt++;
				}
				else if (pflag[p] == TYPEAIRSOLO || pflag[p] == TYPEAIR)
				{
					gairexist += pmass[p];
					apcnt++;
				}
			}
		}

		//2. 如果需要释放流体粒子中溶解的气体形成或增大气泡
		float maxradius = 1.5f*dparam.cellsize.x;
		float maxmass = getMassfromR(maxradius);
		float massaddlimit = 3.0f*dparam.airm0;	//每个气体粒子最多增加3个单位质量
		float addmass;
		if (gcontain>gsolubility)
		{
			//todo: 参数
			if (abs(gcontain - gsolubility) < 2.5*airparticlemass0/*1.3f*gsolubility*/)	//如果相差不大，不进行调整
				return;
			//2.1: 增大已有气泡的体积到最大
			float needadd = gcontain - gsolubility;
			if (apcnt>0)
			{
				for (int di = 0; di <= 1; di++) for (int dj = 0; dj <= 1; dj++) for (int dk = 0; dk <= 1; dk++)
				{
					if (needadd <= 0) break;
					gidx = getidx(i + di, j + dj, k + dk);
					if (gridstart[gidx] == CELL_UNDEF)	continue;
					//	gpos=make_float3(i+di,j+dj,k+dk)*dparam.cellsize.x;
					for (int p = gridstart[gidx]; p<gridend[gidx]; p++)
					{
						if (pflag[p] == TYPEAIRSOLO || pflag[p] == TYPEAIR)
						{
							addmass = min(massaddlimit, maxmass - pmass[p]);
							addmass = max(0.0f, min(needadd, addmass));
							needadd -= addmass;	//有一定的误差
							pmass[p] += addmass;
							if (needadd <= 0)
								break;
						}
					}
				}
			}

			//2.3: 调整每个流体粒子里的气体含量
			float actualadd = gcontain - gsolubility - needadd, eachchange;
			for (int di = 0; di <= 1; di++) for (int dj = 0; dj <= 1; dj++) for (int dk = 0; dk <= 1; dk++)
			{
				if (actualadd <= 0) break;
				gidx = getidx(i + di, j + dj, k + dk);
				if (gridstart[gidx] == CELL_UNDEF)	continue;
				for (int p = gridstart[gidx]; p<gridend[gidx]; p++)
				{
					if (actualadd <= 0) break;
					if (pflag[p] == TYPEFLUID)
					{
						if (paircontain[p] - psolubility[p]>0)
						{
							eachchange = min(actualadd, paircontain[p] - psolubility[p]);
							paircontain[p] -= eachchange;
							actualadd -= eachchange;
						}
					}
				}
			}
		}	//end if( gcontain>gsolubility )
		else if (gairexist>0)		//3: 如果需要吸收气体，且有气体粒子在本网格内
		{
			//todo: 参数
			if (abs(gcontain - gsolubility) < 3.6f*airparticlemass0/*1.3f*gsolubility*/)	//如果相差不大，不进行调整
				return;

			//3.1: 减少气体粒子的质量
			float needminus = gsolubility - gcontain;	//可以吸收的气体量
			float masschangesum = 0;		//实际吸收的气体量
			if (gairexist<needminus)
				needminus = gairexist;
			if (needminus>0)//minus some of them to 0 mass, use another kernel to delete it.
			{
				for (int di = 0; di <= 1; di++) for (int dj = 0; dj <= 1; dj++) for (int dk = 0; dk <= 1; dk++)
				{
					if (needminus <= 0) break;
					gidx = getidx(i + di, j + dj, k + dk);
					if (gridstart[gidx] == CELL_UNDEF)	continue;
					for (int p = gridstart[gidx]; p<gridend[gidx] && needminus>0; p++)
					{
						if (pflag[p] == TYPEAIRSOLO || pflag[p] == TYPEAIR)
						{
							float masschange = min(pmass[p], needminus);	//本气体粒子会被吸收多少质量
							pmass[p] -= masschange;
							needminus -= masschange;
							masschangesum += masschange;
						}
					}
				}

			}
			//3.2: 调整流体粒子中溶解的气体含量. change the fluid particls.
			for (int di = 0; di <= 1; di++) for (int dj = 0; dj <= 1; dj++) for (int dk = 0; dk <= 1; dk++)
			{
				if (masschangesum <= 0) break;
				gidx = getidx(i + di, j + dj, k + dk);
				if (gridstart[gidx] == CELL_UNDEF)	continue;
				for (int p = gridstart[gidx]; p<gridend[gidx] && masschangesum>0; p++)
				{
					if (pflag[p] == TYPEFLUID)
					{
						float containchange = min(max(0.0f, psolubility[p] - paircontain[p]), masschangesum);		//本流体粒子会被填充多少气体量
						paircontain[p] += containchange;
						masschangesum -= containchange;
					}
				}
			}
		}
	}
}

//使用预计算好的位置根据温度和溶解度生成empty气泡，当气泡大于一定体积时，生成AIR粒子。
//对其它模块的影响：markgrid, correctpos, heattransfer.
__global__ void updateEmptyBubbles(float3 *pepos, float3 *pedir, float *peradius, int penum, float3 *parpos, float3 *parvel, float *parmass, float* parTemperature,
	char *parflag, float *parsolubility, float *paraircontain, int parnum, int *addparnums, uint *gridstart, uint *gridend, farray gTemperature)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<penum)
	{
		int airRscale = 2;
		
		float3 ipos = pepos[idx], idir = pedir[idx];
		float iradius = peradius[idx];
		float rthresholdleave = 1.0f*dparam.cellsize.x;	//todo:	//到此半径则转化成实际气体并离开固壁     控制气泡半径
		float rthreshold = max(0.0f, iradius + 0.1f*dparam.cellsize.x);	//此次气泡最大半径，防止突然变大带来的不稳定
		rthreshold = min(rthreshold, rthresholdleave);
		int i, j, k;
		getijkfrompos(i, j, k, ipos);

		//收集需要管的范围内的气体含量，增大体积
		float massorigin = dparam.waterrho * 4 / 3 * M_PI*(pow(iradius, 3))*0.5;
		float masscantake = dparam.waterrho * 4 / 3 * M_PI*(pow(rthreshold, 3) - pow(iradius, 3))*0.5, massadd = 0;	//todo

		int range = 2;
		for (int di = -range; di <= range &&masscantake>0; di++)	for (int dj = -range; dj <= range&&masscantake>0; dj++)	for (int dk = -range; dk <= range&&masscantake>0; dk++)
		if (verifycellidx(i + di, j + dj, k + dk))
		{
			int grididx = getidx(i, j, k);
			for (uint p = gridstart[grididx]; p<gridend[grididx] && masscantake>0; p++)	//遍历所有流体粒子
			{
				if (parflag[p] != TYPEFLUID)
					continue;
				float gasreslease = max(0.0f, paraircontain[p] - parsolubility[p]);
				if (gasreslease <= 0)
					continue;
				gasreslease = min(gasreslease, masscantake);
				massadd += gasreslease;
				masscantake -= gasreslease;
				//paraircontain[p] -= gasreslease;
			}
		}

		float newiradius = pow((massadd + massorigin) / dparam.waterrho / 4 * 3 / M_PI, 1.0 / 3);
		ipos += (newiradius - iradius)*idir;
		float ss = dparam.samplespace;
		if (newiradius + 1e-5 >= rthresholdleave)	//生成实际的气体粒子 
		{
			int num = ceil(newiradius / ss);
			for (float x = -num*ss; x <= newiradius; x += ss)for (float y = -num*ss; y <= newiradius; y += ss)for (float z = -num*ss; z <= newiradius; z += ss)
			{
				if (x*x + y*y + z*z>newiradius*newiradius)
					continue;
				int addindex = atomicAdd(&addparnums[0], 1) + parnum;
				parmass[addindex] = dparam.airm0;	//todo:
				parpos[addindex] = ipos + make_float3(x, y, z);
				parflag[addindex] = TYPEAIR;
				parvel[addindex] = make_float3(0.0f);
				parTemperature[addindex] = gTemperature[getidx(i, j, 1)]; //todo: 找到当前气泡最下面网格的温度
				paraircontain[addindex] = 0.0f;
				parsolubility[addindex] = 0.0f;
			}
			ipos.z = 1.1f*dparam.cellsize.x;	//重置位置
			newiradius = 0;
		}
		peradius[idx] = newiradius;
		pepos[idx] = ipos;
	}
}

__device__ void mat4_mul(matrix4* dst, const matrix4* m0, const matrix4* m1)
{
	int row;
	int col;
	int i;


	for (row = 0; row < 4; row++)
	for (col = 0; col < 4; col++)
	for (i = 0; i < 4; i++)
		dst->m[row * 4 + col] += m0->m[row * 4 + i] * m1->m[i * 4 + col];

}
__device__ void mat4_mulvec3_as_mat3(float3* dst, const matrix4* m, const float3* v)
{
	float new_x;
	float new_y;
	float new_z;

	new_x = v->x*m->m[0 + 4 * 0] + v->y*m->m[0 + 4 * 1] + v->z*m->m[0 + 4 * 2];
	new_y = v->x*m->m[1 + 4 * 0] + v->y*m->m[1 + 4 * 1] + v->z*m->m[1 + 4 * 2];
	new_z = v->x*m->m[2 + 4 * 0] + v->y*m->m[2 + 4 * 1] + v->z*m->m[2 + 4 * 2];
	dst->x = new_x;
	dst->y = new_y;
	dst->z = new_z;
}

__global__ void MeltingSolidByHeat(float *pTemperature, float *pLHeat, char *pflag, int pnum, float LiquidHeatTh, float meltTemperature, int *numchange)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum && pflag[idx] == TYPESOLID)
	{
		//if( pTemperature[idx]>meltTemperature )
		if (pLHeat[idx]>LiquidHeatTh)
		{
			pflag[idx] = TYPEFLUID;
			pLHeat[idx] = LiquidHeatTh;
			atomicAdd(&numchange[0], 1);
		}
	}
}

__global__ void FreezingSolidByHeat(float3* ppos, float *pLHeat, char *pflag, int pnum, int *numchange, uint *gridstart, uint *gridend)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum && pflag[idx] == TYPEFLUID)
	{
		//if( pTemperature[idx]>meltTemperature )
		if (pLHeat[idx]<0)
		{
			//determine a new position which is appropriate for solid.
			//找距离最近的固体粒子
			int i, j, k;
			float3 ipos = ppos[idx];
			getijkfrompos(i, j, k, ipos);
			float mindis = 1000;
			int minidx = -1;
			int width = 1;
			int cntsolid = 0;
			float h = dparam.cellsize.x;
			for (int di = -width; di <= width; di++) for (int dj = -width; dj <= width; dj++) for (int dk = -width; dk <= width; dk++)
			if (verifycellidx(i + di, j + dj, k + dk))
			{
				int gidx = getidx(i + di, j + dj, k + dk);
				uint start = gridstart[gidx];
				if (start == CELL_UNDEF) continue;
				for (int p = start; p<gridend[gidx]; p++)
				{
					if (pflag[p] == TYPESOLID)
					{
						float dis = length(ppos[p] - ipos);
						if (dis< h)
							cntsolid++;
						if (length(ppos[p] - ipos)<mindis)
							mindis = length(ppos[p] - ipos), minidx = p;
					}
				}
			}

			if (minidx != -1 && mindis<dparam.cellsize.x && cntsolid>2)//周围有固体粒子
			{
				pflag[idx] = TYPESOLID;
				pLHeat[idx] = 0;
				atomicAdd(&numchange[0], 1);
				if (mindis > dparam.samplespace)
				{
					ipos = normalize(ipos - ppos[minidx])*dparam.samplespace + ppos[minidx];
					ppos[idx] = ipos;
				}
			}
		}
	}
}

//计算air solo particle与流体场之间的drag force，直接在本函数里修改了速度。以dragparam为影响大小的参数。
__global__ void calDragForce(float3 *ppos, float3 *pvel, char *pflag, int pnum, farray ux, farray uy, farray uz, float dragparamsolo, float dragparamgrid, SCENE scene)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] != TYPEAIRSOLO)
			return;

		float3 ipos = ppos[idx], ivel = pvel[idx];
		//compute the grid index
		int i, j, k;
		getijkfrompos(i, j, k, ipos);

		//compute drag "force" (actually not "force", is velocity change, tuning alpha is very important)
		float3 gridvel = getParticleVelFromGrid(ipos, ux, uy, uz);
		float3 gridpos = make_float3(i, j, k);
		float3 dragf_b = dragparamsolo * length(gridvel - ivel) * (gridvel - ivel);		//指向grid's velocity，施加给bubble的，质量被系统归一成1
		/*	float alpha = 0.5f;*/
		float3 velChange_g = -dragf_b*dragparamgrid*dparam.dt;		//施加给网格的，要增加一个比例系数，因为同样受力的情况下，网格的质量大，速度改变要小一些

		//update for grid
		float ux0, ux1, uy0, uy1, uz0, uz1;
		float3 weight = ipos / dparam.cellsize.x - gridpos;		//权重 in [0-1]
		ux0 = velChange_g.x*(1 - weight.x), ux1 = velChange_g.x*weight.x;
		uy0 = velChange_g.y*(1 - weight.y), uy1 = velChange_g.y*weight.y;
		uz0 = velChange_g.z*(1 - weight.z), uz1 = velChange_g.z*weight.z;

		atomicAdd(&(ux.data[getidx(i, j, k, NX + 1, NY, NZ)]), ux0);
		atomicAdd(&(ux.data[getidx(i + 1, j, k, NX + 1, NY, NZ)]), ux1);
		atomicAdd(&(uy.data[getidx(i, j, k, NX, NY + 1, NZ)]), uy0);
		atomicAdd(&(uy.data[getidx(i, j + 1, k, NX, NY + 1, NZ)]), uy1);
		atomicAdd(&(uz.data[getidx(i, j, k, NX, NY, NZ + 1)]), uz0);
		atomicAdd(&(uz.data[getidx(i, j, k + 1, NX, NY, NZ + 1)]), uz1);

		//update for particle，注意是需要反向的。todo：只在Interaction场景里用？
		if (scene == SCENE_INTERACTION || scene == SCENE_INTERACTION_HIGHRES)
			pvel[idx] += dragf_b*dparam.dt;
	}
}

__global__ void accumulate_GPU_k(int num, float3* out, float3* a)//dsum, a.data, flag, n
{
	extern __shared__ float3 ddata[];

	uint tid = threadIdx.x;
	uint i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;

	ddata[tid] = (i >= num) ? make_float3(0, 0, 0) : a[i];
	__syncthreads();

	for (int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid<s)
			ddata[tid] += ddata[tid + s];
		__syncthreads();
	}

	if (tid == 0)
		out[blockIdx.x] = ddata[0];
}

__global__ void compute_cI_k(int pnum, char* parflag, float3 *parPos, float3 *parVel, float3* c, float3* weight, float3 rg)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;

	if (idx<pnum)
	{
		if (parflag[idx] == TYPESOLID)
		{
			float dis = length(parPos[idx] - rg);
			if (dis>1e-6)
			{
				c[idx] = cross(parPos[idx] - rg, parVel[idx]);
				weight[idx] = make_float3(dis, 0, 0);
			}
			else
				c[idx] = weight[idx] = make_float3(0);
		}
		else
		{
			c[idx] = weight[idx] = make_float3(0);
			//c[idx] = make_float3(0,0,0);
		}
	}
}

__global__ void setVelZeroSolid_k(float3 *parvel, char *parflag, int pnum)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum && parflag[idx] == TYPESOLID)
		parvel[idx] = make_float3(0);
}

__global__ void computeVelSolid_k(float3* parPos, char* parflag, float3* parVel, int pnum, float3 rg, float3 R, float3 T)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum && parflag[idx] == TYPESOLID)
	{

		float3 v_half = cross(R, parPos[idx] - rg);		//粒子的角速度`
		v_half += T;								//固体粒子的总速度
		v_half = 0.5*(parVel[idx] + v_half);
		parVel[idx] = v_half;
		//	parVel[idx] = make_float3(0);
	}
}

__device__ inline float3 transposeParticle(float3 p, matrix3x3 rm)
{
	float3 res;
	res.x = p.x*rm.x00 + p.y*rm.x10 + p.z*rm.x20;
	res.y = p.x*rm.x01 + p.y*rm.x11 + p.z*rm.x21;
	res.z = p.x*rm.x02 + p.y*rm.x12 + p.z*rm.x22;
	return res;
}
//由rotation matrix "rm"来计算各粒子的位置
__global__ void computePosSolid_k(float3* parPos, char* parflag, int pnum, float3 rg, float3 rg0, matrix3x3 rm)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum && parflag[idx] == TYPESOLID)
	{
		float3 transp = parPos[idx] - rg0;
		transp = transposeParticle(transp, rm);
		parPos[idx] = transp + rg;
	}
}

__global__ void computeSolidVertex_k(float3* vertexpos, int vnum, float3 rg, float3 rg0, matrix3x3 rm)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<vnum)
	{
		float3 transp = vertexpos[idx] - rg0;
		transp = transposeParticle(transp, rm);
		vertexpos[idx] = transp + rg;
	}
}

__global__ void set_nonsolid_2_zero(char* pflag, int pnum, float3* Pos, float3* Vel)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum && pflag[idx] != TYPESOLID)
	{
		Pos[idx] = make_float3(0, 0, 0);
		Vel[idx] = make_float3(0, 0, 0);
	}
}

//在粒子层面处理fluid, air, airsolo粒子与solid的碰撞关系，保证不会穿过边界到solid的内部。
__global__ void CollisionWithSolid_k(float3 *ppos, float3 *pvel, char *pflag, int pnum, farray phisolid, farray sux, farray suy, farray suz, SCENE scene, float bounceVelParam, float bouncePosParam)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] == TYPESOLID)
			return;
		float3 ipos = ppos[idx];
		float3 ivel = pvel[idx];
		float iphi = getScaleFromFrid(ipos, phisolid);
		if (iphi <= 0.5f)		//靠近固体，距离只有半个格子
		{
			float3 svel = getParticleVelFromGrid(ipos, sux, suy, suz);
			float3 rvel = ivel - svel;
			float d = dparam.cellsize.x * 0.5f;
			float3 phigrad;
			phigrad.x = getScaleFromFrid(ipos + make_float3(d, 0, 0), phisolid) - getScaleFromFrid(ipos - make_float3(d, 0, 0), phisolid);
			phigrad.y = getScaleFromFrid(ipos + make_float3(0, d, 0), phisolid) - getScaleFromFrid(ipos - make_float3(0, d, 0), phisolid);
			phigrad.z = getScaleFromFrid(ipos + make_float3(0, 0, d), phisolid) - getScaleFromFrid(ipos - make_float3(0, 0, d), phisolid);
			if (length(phigrad) > 0)
			{
				phigrad = normalize(phigrad);		//指向外侧
				if (dot(rvel, phigrad)<0 || scene == SCENE_FREEZING)	//相对速度指向内侧
				{
					ivel -= bounceVelParam * dot(rvel, phigrad)*phigrad;		//法向速度置为与固体一样
					if (scene == SCENE_FREEZING)
						ivel -= 0.1f* (rvel - dot(rvel, phigrad)*phigrad);		//切向速度
				}
				ipos += bouncePosParam * phigrad * (0.5f - iphi) * dparam.cellsize.x;
			}
		}
		//并根据新的速度更新位置
		ipos += ivel*dparam.dt;
		//边界
		float rate = 0.5f, ratevel = -0.5f;
		if (pflag[idx] == TYPEAIRSOLO)
			rate = 0.8f, ratevel = -0.5f;
		float3 tmin = dparam.gmin + (dparam.cellsize + make_float3(rate*dparam.cellsize.x));
		float3 tmax = dparam.gmax - (dparam.cellsize + make_float3(rate*dparam.cellsize.x));
		// 		if( ipos.x>tmax.x )
		// 			ivel.x *=ratevel, ipos.x=tmax.x;
		// 		if( ipos.x<tmin.x )
		// 			ivel.x *= ratevel, ipos.x=tmin.x;
		// 		if( ipos.y>tmax.y )
		// 			ivel.y *=ratevel, ipos.y=tmax.y;
		// 		if( ipos.y<tmin.y )
		// 			ivel.y *= ratevel, ipos.y=tmin.y;
		// 		if( ipos.z>tmax.z )
		// 			ivel.z *=ratevel, ipos.z=tmax.z;
		// 		if( ipos.z<tmin.z )
		// 			ivel.z *= ratevel, ipos.z=tmin.z;
		if (ipos.x <= tmin.x)
			ipos.x = tmin.x, ivel.x = 0.0f;
		if (ipos.y <= tmin.y)
			ipos.y = tmin.y, ivel.y = 0.0f;
		if (ipos.z <= tmin.z)
			ipos.z = tmin.z, ivel.z = 0.0f;

		if (ipos.x >= tmax.x)
			ipos.x = tmax.x, ivel.x = 0.0f;
		if (ipos.y >= tmax.y)
			ipos.y = tmax.y, ivel.y = 0.0f;
		if (ipos.z >= tmax.z)
			ipos.z = tmax.z, ivel.z = 0.0f;

		//存储新的速度和位置
		pvel[idx] = ivel;
		ppos[idx] = ipos;
	}
}

//专门为melting and freezing场景写的，粒度要更细一些。在粒子层面处理fluid, air, airsolo粒子与solid的碰撞关系，保证不会穿过边界到solid的内部。
__global__ void CollisionWithSolid_Freezing(float3 *ppos, float3 *pvel, char *pflag, int pnum, farray phisolid, uint* gridstart, uint* gridend)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (pflag[idx] == TYPESOLID)
			return;

		float3 ipos = ppos[idx];
		float3 ivel = pvel[idx];
		int i, j, k;
		getijkfrompos(i, j, k, ipos);

		float iphi = getScaleFromFrid(ipos, phisolid);
		if (iphi <= 1.0f)		//有发生碰撞的可能，再进行检测
		{
			float r = 0.25f*dparam.cellsize.x;
			float3 collisionpos = make_float3(0), dir;
			float depth = 0, dis, adhesionDis = 0;
			int cntcollide = 0, cntadhesion = 0;
			float h = 4 * r;
			for (int di = -1; di <= 1; di++)for (int dj = -1; dj <= 1; dj++)for (int dk = -1; dk <= 1; dk++)
			{
				if (verifycellidx(i + di, j + dj, k + dk))
				{
					int grididx = getidx(i + di, j + dj, k + dk);
					int start = gridstart[grididx];
					if (start == CELL_UNDEF)
						continue;
					for (uint p = start; p<gridend[grididx]; p++)
					{
						dir = ipos - ppos[p];
						dis = length(dir);
						if (dis>0 && dis<2 * r)	//碰撞
						{
							collisionpos += ppos[p];
							depth = max(depth, 2 * r - dis);
							cntcollide++;
						}
						else if (dis< h)
						{
							adhesionDis += dis;
							cntadhesion++;
						}
					}
				}
			}
			float3 n;
			float d = dparam.cellsize.x * 0.5f;
			n.x = getScaleFromFrid(ipos + make_float3(d, 0, 0), phisolid) - getScaleFromFrid(ipos - make_float3(d, 0, 0), phisolid);
			n.y = getScaleFromFrid(ipos + make_float3(0, d, 0), phisolid) - getScaleFromFrid(ipos - make_float3(0, d, 0), phisolid);
			n.z = getScaleFromFrid(ipos + make_float3(0, 0, d), phisolid) - getScaleFromFrid(ipos - make_float3(0, 0, d), phisolid);
			float3 originalvel = ivel;
			if (length(n) > 0)
			{
				n = normalize(n);		//指向外侧
				if (cntcollide>0)	//发生碰撞
				{
					collisionpos /= cntcollide;

					if (length(n) > 0)
					{
						//correct vel and pos;
						ivel -= dot(originalvel, n)*n;		//法向速度置为与固体一样
						//ivel *= 1.1f;
						ipos += depth * n;
					}
				}
				else if (cntadhesion>0)		//有一定的吸引力
				{
					float alpha = 0.1f;
					ivel -= n * alpha * length(ivel);
				}
			}

		}

		//并根据新的速度更新位置
		//边界
		float3 tmin = dparam.gmin + (dparam.cellsize + make_float3(0.3f*dparam.samplespace));
		float3 tmax = dparam.gmax - (dparam.cellsize + make_float3(0.3f*dparam.samplespace));
		if (ipos.x>tmax.x)
			ivel.x *= -0.5f, ipos.x = tmax.x;
		if (ipos.x<tmin.x)
			ivel.x *= -0.5f, ipos.x = tmin.x;
		if (ipos.y>tmax.y)
			ivel.y *= -0.5f, ipos.y = tmax.y;
		if (ipos.y<tmin.y)
			ivel.y *= -0.5f, ipos.y = tmin.y;
		if (ipos.z>tmax.z)
			ivel.z *= -0.5f, ipos.z = tmax.z;
		if (ipos.z<tmin.z)
			ivel.z *= -0.5f, ipos.z = tmin.z;

		ipos += ivel*dparam.dt;
		//存储新的速度和位置
		pvel[idx] = ivel;
		ppos[idx] = ipos;
	}
}

__global__ void buoyancyForSolid(float3 *ppos, float3 *pvel, char *pflag, int pnum, uint *gridstart, uint *gridend, float SolidBuoyanceParam)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum && pflag[idx] == TYPESOLID)
	{
		int cnt = 0;
		int i, j, k;
		float3 ipos = ppos[idx];
		getijkfrompos(i, j, k, ipos);
		float r = dparam.cellsize.x;
		for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++)
		{
			if (verifycellidx(i + di, j + dj, k + dk))
			{
				int gidx = getidx(i + di, j + dj, k + dk);
				uint start = gridstart[gidx];
				if (start != CELL_UNDEF)
				{
					for (uint p = start; p<gridend[gidx]; p++)
					if (pflag[p] == TYPEFLUID && length(ppos[p] - ipos)<r)
						cnt++;
				}
			}
		}
		if (cnt>2)
			pvel[idx].z += (dparam.waterrho - dparam.solidrho) * SolidBuoyanceParam * dparam.dt;
	}
}

__global__ void solidCollisionWithBound(float3 *ppos, float3 *pvel, char *pflag, int pnum, float SolidbounceParam, int nSolPoint)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pnum && pflag[idx] == TYPESOLID)
	{
		//check position
		float3 tmin = dparam.gmin + (dparam.cellsize + make_float3(0.3f*dparam.samplespace));
		float3 tmax = dparam.gmax - (dparam.cellsize + make_float3(0.3f*dparam.samplespace));
		float3 ipos = ppos[idx];
		float3 ivel = pvel[idx];

		//float eps=1e-6;
		// 反向的速度与“穿透深度，系数，粒子个数”相关。
		//（与粒子个数相关主要是因为这个速度是起到“惩罚力”的作用，而粒子个数起到“质量”的作用，在粒子的速度向刚体转换的时候，相当于一个“平均(除质量)”的过程）
		if (ipos.x<tmin.x)
			ivel.x += (tmin.x - ipos.x) * SolidbounceParam * nSolPoint;
		if (ipos.x>tmax.x)
			ivel.x -= (ipos.x - tmax.x) * SolidbounceParam * nSolPoint;
		if (ipos.y<tmin.y)
			ivel.y += (tmin.y - ipos.y) * SolidbounceParam * nSolPoint;
		if (ipos.y>tmax.y)
			ivel.y -= (ipos.y - tmax.y) * SolidbounceParam * nSolPoint;
		if (ipos.z<tmin.z)
			ivel.z += (tmin.z - ipos.z) * SolidbounceParam * nSolPoint;
		if (ipos.z>tmax.z)
			ivel.z -= (ipos.z - tmax.z) * SolidbounceParam * nSolPoint;

		pvel[idx] = ivel;
		//ppos[idx]=ipos;	//不能修改位置，刚体会变形
	}
}


//there is a problem here, remember to solve it.
// __global__ void genAirFromSolid_k( float3 *ppos, float3 *pvel, char *pflag, float *psolubility, float *paircontain, float *pmass, float *pTemperature,int pnum, 
// 								charray lsmark, farray phisolid, farray Tgrid, int *addnum, float *randfloat, int nrandnum, int frame )
// {
// 	int idx=__mul24( blockIdx.x, blockDim.x )+threadIdx.x;
// 	if( idx<dparam.gnum &&lsmark[idx]==TYPEFLUID && phisolid[idx]>0 )	//此格子是流体格子
// 	{
// 		int i,j,k;
// 		getijk( i,j,k,idx);
// 		bool flag=false;
// 		for( int di=-1; di<=1; di++ )		for( int dj=-1; dj<=1; dj++ )		for( int dk=-1; dk<=1; dk++ )
// 		{
// 			if(verifycellidx(i+di,j+dj,k+dk) && phisolid( i+di,j+dj,k+dk)<0 )
// 				flag=true;
// 		}
// 		if( !flag )
// 			return;
// 
// 		int cnt= (idx*frame) % ( nrandnum-100 );
// 		if( randfloat[cnt++]>0.95 )	//if randnum>thresold, generate a airsolo bubble
// 		{
// 			int addidx=atomicAdd( addnum, 1 );
// 			float3 addpos= (make_float3(randfloat[cnt], randfloat[cnt], randfloat[cnt])  + make_float3(i,j,k) ) * dparam.cellsize.x;
// 			ppos[pnum+addidx] = addpos;
// 			pvel[pnum+addidx]=make_float3(0);
// 			pflag[pnum+addidx]=TYPEAIRSOLO;
// 			psolubility[pnum+addidx]=0;
// 			paircontain[pnum+addidx]=0;
// 			pmass[pnum+addidx]=dparam.airm0;
// 			pTemperature[pnum+addidx]=getScaleFromFrid( addpos, Tgrid );
// 		}
// 	}
// }

//这个函数是考虑latent heat的主函数，当温度超过界限时(如固体的温度高于熔点)，则多余的热量放到latent heat里；当latent heat满足一定条件时，发生phase change.
__global__ void updateLatentHeat_k(float *parTemperature, float *parLHeat, char *partype, int pnum, float meltingpoint, float boilingpoint, float LiquidHeatTh)
{
	int idx = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (idx<pnum)
	{
		if (partype[idx] == TYPESOLID && parTemperature[idx]>meltingpoint)
		{
			parLHeat[idx] += parTemperature[idx] - meltingpoint;
			parTemperature[idx] = meltingpoint;
		}
		if (partype[idx] == TYPEFLUID)
		{
			if (parTemperature[idx]<meltingpoint)
			{
				parLHeat[idx] -= meltingpoint - parTemperature[idx];
				parTemperature[idx] = meltingpoint;
			}
			else if (parTemperature[idx]>boilingpoint)
			{
				parLHeat[idx] += parTemperature[idx] - boilingpoint;
				//	parLHeat[idx] = min( parLHeat[idx], LiquidHeatTh+5 );
				parTemperature[idx] = boilingpoint;
			}
			else
				parLHeat[idx] = LiquidHeatTh;
		}
	}
}

__global__ void pouringwater(float3* pos, float3* vel, float* parmass, char* parflag, float *ptemperature, float *pLHeat, float *pGasContain, int parnum,
	float3 *ppourpos, float3 *ppourvel, char pourflag, int pournum, float *randfloat, int randnum, int frame, float posrandparam, float velrandparam,
	float defaultLiquidT, float LiquidHeatTh)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<pournum)
	{
		//速度与位置的随机化
		int randbase = (frame + idx) % (randnum - 6);
		float3 randvel = make_float3(randfloat[randbase], randfloat[randbase + 1], randfloat[randbase + 2]) *2.0f - 1.0f;
		randbase += 3;
		float3 randpos = make_float3(randfloat[randbase], randfloat[randbase + 1], randfloat[randbase + 2]) *2.0f - 1.0f;

		pos[parnum + idx] = ppourpos[idx] + randpos * posrandparam*dparam.samplespace;
		vel[parnum + idx] = ppourvel[idx] + randvel * velrandparam;
		parmass[parnum + idx] = dparam.m0;
		parflag[parnum + idx] = pourflag;
		ptemperature[parnum + idx] = defaultLiquidT;
		pLHeat[parnum + idx] = LiquidHeatTh;
		pGasContain[parnum + idx] = 0;
	}
}


inline __device__ float getlen(float x, float y)
{
	return sqrt(x*x + y*y);
}
__global__ void initheat_grid_k(farray tp, charray mark)
{
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx<dparam.gnum)
	{
		int i, j, k;
		getijk(i, j, k, idx);
		float x = i, z = k;
		float r = NX*0.15;

		if (getlen(x - NX / 4, z - NZ / 4) <= r)
			tp[idx] = 100, mark[idx] = TYPESOLID;
		else if (getlen(x - NX / 4 * 3, z - NZ / 4 * 3) <= r)
			tp[idx] = 0, mark[idx] = TYPEFLUID;
		else if (z<NZ / 2)
			tp[idx] = 20, mark[idx] = TYPEVACUUM;
		else
			tp[idx] = 80, mark[idx] = TYPEAIR;
	}
}