#include <Windows.h>
#include<cuda.h>
#include<helper_cuda.h>
#include"helper_math.h"
#include<helper_functions.h>
#include "glew.h"
#include<cuda_gl_interop.h>
#include "spray.h"
#include "spray_k.cuh"
#include "timer.h"
#include <cuda_runtime.h>
# include < curand.h >
#include "radixsort.cuh"

extern void printTime(bool btime, char* info, CTimer &time);

inline int getidx(int i, int j, int k)
{
	return (i*NZ*NY + j*NZ + k);
}


void cspray::initmem_bubble()
{
	copyparamtoGPU(hparam);
	copyNXNYNZtoGPU(NX, NY, NZ);

	//press
	int gsmemsize = sizeof(float)*hparam.gnum;
	cudaMalloc((void**)&mpress.data, gsmemsize);
	cudaMalloc((void**)&temppress.data, gsmemsize);
	cudaMemset(mpress.data, 0, gsmemsize);
	cudaMemset(temppress.data, 0, gsmemsize);

	//div
	cudaMalloc((void**)&mDiv.data, gsmemsize);
	cudaMemset(mDiv.data, 0, gsmemsize);

	//phi
	cudaMalloc((void**)&phifluid.data, gsmemsize);
	cudaMemset(phifluid.data, 0, gsmemsize);
	cudaMalloc((void**)&phiair.data, gsmemsize);
	cudaMemset(phiair.data, 0, gsmemsize);
	cudaMalloc((void**)&phisolid.data, gsmemsize);
	cudaMemset(phisolid.data, 0, gsmemsize);

	//level set value.
	cudaMalloc((void**)&lsair.data, gsmemsize);
	cudaMalloc((void**)&lsfluid.data, gsmemsize);
	cudaMalloc((void**)&lsmerge.data, gsmemsize);
	//gradient of level set
	cudaMalloc((void**)&phigrax.data, gsmemsize);
	cudaMalloc((void**)&phigray.data, gsmemsize);
	cudaMalloc((void**)&phigraz.data, gsmemsize);
	cudaMalloc((void**)&phigrax_air.data, gsmemsize);
	cudaMalloc((void**)&phigray_air.data, gsmemsize);
	cudaMalloc((void**)&phigraz_air.data, gsmemsize);

	//surface tension
	cudaMalloc((void**)&surfacetension.data, gsmemsize);

	//u
	int gvxmemsize = sizeof(float)*hparam.gvnum.x;
	int gvymemsize = sizeof(float)*hparam.gvnum.y;
	int gvzmemsize = sizeof(float)*hparam.gvnum.z;
	cudaMalloc((void**)&waterux.data, gvxmemsize);
	waterux.setdim(NX + 1, NY, NZ);
	cudaMalloc((void**)&wateruy.data, gvymemsize);
	wateruy.setdim(NX, NY + 1, NZ);
	cudaMalloc((void**)&wateruz.data, gvzmemsize);
	wateruz.setdim(NX, NY, NZ + 1);

	cudaMalloc((void**)&waterux_old.data, gvxmemsize);
	waterux_old.setdim(NX + 1, NY, NZ);
	cudaMalloc((void**)&wateruy_old.data, gvymemsize);
	wateruy_old.setdim(NX, NY + 1, NZ);
	cudaMalloc((void**)&wateruz_old.data, gvzmemsize);
	wateruz_old.setdim(NX, NY, NZ + 1);

	cudaMalloc((void**)&tmpux.data, gvxmemsize);
	tmpux.setdim(NX + 1, NY, NZ);
	cudaMalloc((void**)&tmpuy.data, gvymemsize);
	tmpuy.setdim(NX, NY + 1, NZ);
	cudaMalloc((void**)&tmpuz.data, gvzmemsize);
	tmpuz.setdim(NX, NY, NZ + 1);

	cudaMalloc((void**)&solidux.data, gvxmemsize);
	solidux.setdim(NX + 1, NY, NZ);
	cudaMalloc((void**)&soliduy.data, gvymemsize);
	soliduy.setdim(NX, NY + 1, NZ);
	cudaMalloc((void**)&soliduz.data, gvzmemsize);
	soliduz.setdim(NX, NY, NZ + 1);

	cudaMemset(waterux.data, 0, gvxmemsize);
	cudaMemset(wateruy.data, 0, gvymemsize);
	cudaMemset(wateruz.data, 0, gvzmemsize);
	cudaMemset(waterux_old.data, 0, gvxmemsize);
	cudaMemset(wateruy_old.data, 0, gvymemsize);
	cudaMemset(wateruz_old.data, 0, gvzmemsize);

	//for air u
	{
		cudaMalloc((void**)&airux.data, gvxmemsize);
		airux.setdim(NX + 1, NY, NZ);
		cudaMalloc((void**)&airuy.data, gvymemsize);
		airuy.setdim(NX, NY + 1, NZ);
		cudaMalloc((void**)&airuz.data, gvzmemsize);
		airuz.setdim(NX, NY, NZ + 1);

		cudaMalloc((void**)&airux_old.data, gvxmemsize);
		airux_old.setdim(NX + 1, NY, NZ);
		cudaMalloc((void**)&airuy_old.data, gvymemsize);
		airuy_old.setdim(NX, NY + 1, NZ);
		cudaMalloc((void**)&airuz_old.data, gvzmemsize);
		airuz_old.setdim(NX, NY, NZ + 1);

		cudaMemset(airux.data, 0, gvxmemsize);
		cudaMemset(airuy.data, 0, gvymemsize);
		cudaMemset(airuz.data, 0, gvzmemsize);
		cudaMemset(airux_old.data, 0, gvxmemsize);
		cudaMemset(airuy_old.data, 0, gvymemsize);
		cudaMemset(airuz_old.data, 0, gvzmemsize);
	}

	//mark
	cudaMalloc((void**)&mmark, sizeof(char)*hparam.gnum);
	cudaMalloc((void**)&mark_terrain, sizeof(char)*hparam.gnum);

	//particle
	cudaMalloc((void**)&mParPos, parNumMax*sizeof(float3));
	cudaMalloc((void**)&mParVel, parNumMax*sizeof(float3));
	cudaMemset(mParVel, 0, parNumMax*sizeof(float3));
	cudaMalloc((void**)&parflag, parNumMax*sizeof(char));
	cudaMalloc((void**)&parmass, parNumMax*sizeof(float));
	cudaMalloc((void**)&parTemperature, parNumMax*sizeof(float));
	cudaMalloc((void**)&parLHeat, parNumMax*sizeof(float));
	cudaMalloc((void**)&parsolubility, parNumMax*sizeof(float));
	cudaMalloc((void**)&pargascontain, parNumMax*sizeof(float));
	//particle attribute: no need for sort
	cudaMalloc((void**)&pardens, parNumMax*sizeof(float));
	cudaMalloc((void**)&parpress, parNumMax*sizeof(float));

	//GY
	//cudaMalloc((void**)&initialSolPos, parNumMax*sizeof(float3));/////////////////////////////GY
	cudaMalloc((void**)&c, parNumMax*sizeof(float3));
	cudaMalloc((void**)&I, parNumMax*sizeof(float3));
	cudaMalloc((void**)&solidParPos, parNumMax*sizeof(float3));
	cudaMalloc((void**)&solidParVelFLIP, parNumMax*sizeof(float3));
	// 	cudaMemset( c, 0, parNumMax*sizeof(float3));
	// 	cudaMemset( I, 0, parNumMax*sizeof(float3));

	//for deleting particles.
	cudaMalloc((void**)&preservemark, parNumMax*sizeof(uint));
	cudaMalloc((void**)&preservemarkscan, parNumMax*sizeof(uint));

	//sort the particles.
	cudaMalloc((void**)&gridHash, parNumMax*sizeof(uint));
	cudaMalloc((void**)&gridIndex, parNumMax*sizeof(uint));
	cudaMalloc((void**)&gridstart, 8 * hparam.gnum*sizeof(uint));
	cudaMalloc((void**)&gridend, 8 * hparam.gnum*sizeof(uint));
	cudaMalloc((void**)&tmpParPos, parNumMax*sizeof(float3));
	cudaMalloc((void**)&tmpParVelFLIP, parNumMax*sizeof(float3));
	cudaMalloc((void**)&tmpparflag, parNumMax*sizeof(char));
	cudaMalloc((void**)&tmpparmass, parNumMax*sizeof(float));
	cudaMalloc((void**)&tmpparTemperature, parNumMax*sizeof(float));
	cudaMalloc((void**)&tmpparHeat, parNumMax*sizeof(float));
	cudaMalloc((void**)&tempsolubility, parNumMax*sizeof(float));
	cudaMalloc((void**)&tempgascontain, parNumMax*sizeof(float));



	//for pcg
	cudaMalloc((void**)&pre.data, sizeof(float)*hparam.gnum);
	cudaMalloc((void**)&z.data, sizeof(float)*hparam.gnum);
	cudaMalloc((void**)&r.data, sizeof(float)*hparam.gnum);
	cudaMalloc((void**)&p.data, sizeof(float)*hparam.gnum);

	//for rand number of CUDA
	{
		cudaMalloc((void**)&randfloat, sizeof(float)*randfloatcnt);
		curandGenerator_t gen;
		// Create pseudo - random number generator 
		(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
		/* Set seed */
		(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
		//产生0.0~1.0f的随机数
		(curandGenerateUniform(gen, randfloat, randfloatcnt));
	}

	//热传导的温度
	cudaMalloc((void**)&Tp.data, sizeof(float)*hparam.gnum);
	cudaMalloc((void**)&Tp_old.data, sizeof(float)*hparam.gnum);
	cudaMalloc((void**)&Tp_save.data, sizeof(float)*hparam.gnum);
	cudaMalloc((void**)&fixedHeat.data, sizeof(float)*hparam.gnum);

	//seed
	cudaMalloc((void**)&dseedcell, sizeof(int)*seednum);

	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	//for CPU computation.
	hparLHeat = new float[parNumMax];
	hparflag = new char[parNumMax];
	hwaterux.setdim(NX + 1, NY, NZ);
	hwateruy.setdim(NX, NY + 1, NZ);
	hwateruz.setdim(NX, NY, NZ + 1);
	hwaterux.data = new float[hparam.gvnum.x];
	hwateruy.data = new float[hparam.gvnum.y];
	hwateruz.data = new float[hparam.gvnum.z];
	hpos = new float3[parNumMax];
	hvel = new float3[parNumMax];
	hmass = new float[parNumMax];
	hgridstart = new uint[hparam.gnum];
	hgridend = new uint[hparam.gnum];
}

float cspray::checkGridFarray(farray u)
{
	static float *ux2 = new float[gvblocknum*threadnum];
	cudaMemcpy(ux2, u.data, sizeof(float)*u.xn*u.yn*u.zn, cudaMemcpyDeviceToHost);

	float sum = 0.0f;
	for (int idx = 0; idx<u.xn*u.yn*u.zn; idx++)
	{
		int i, j, k;
		getijk(i, j, k, idx, u.xn, u.yn, u.zn);
		if (!(ux2[idx]>-1 || ux2[idx]<1))
		{
			printf("i,j,k=%d,%d,%d, value=%f\n", i, j, k, ux2[idx]);

			break;
		}
		//sum+=abs(ux2[idx]);
	}
	return sum;
}

void cspray::project_CG(farray ux, farray uy, farray uz)
{
	CTimer time;
	time.startTimer();
	cudaMemset(mDiv.data, 0, sizeof(float)*hparam.gnum);
	cptdivergence << <gsblocknum, threadnum >> >(mDiv, ux, uy, uz, mmark);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	//solve pressure
	cudaMemset(mpress.data, 0, sizeof(float)*hparam.gnum);

	solver_cg(mmark, mpress, mDiv, hparam.gnum);

	// 	int cnt=0;
	// 	static char* hmark=new char[hparam.gnum];
	// 	cudaMemcpy( hmark, mmark.data, sizeof(char)*hparam.gnum, cudaMemcpyDeviceToHost);
	// 	for( int i=0; i<hparam.gnum; ++i )
	// 		if( hmark[i]==TYPEFLUID )
	// 			cnt++;
	// 	printf( "typefluid=%d\n", cnt );
	// 
	// 	float tsum = checkGridFarray(mpress);
	// 	printf( "sum of press = %f\n", tsum );

	//compute divergence-free velocity.
	subGradPress << <gvblocknum, threadnum >> >(mpress, ux, uy, uz);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}


void cspray::project_Jacobi(farray ux, farray uy, farray uz)
{
	CTimer time;
	time.startTimer();
	cudaMemset(mDiv.data, 0, sizeof(float)*hparam.gnum);
	cptdivergence << <gsblocknum, threadnum >> >(mDiv, ux, uy, uz, mmark);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	//solve pressure
	cudaMemset(mpress.data, 0, sizeof(float)*hparam.gnum);

	solver_Jacobi(mmark, mpress, mDiv, MAXITER);

	//compute divergence-free velocity.
	subGradPress << <gvblocknum, threadnum >> >(mpress, ux, uy, uz);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}

void cspray::wateradvect()
{
#if 0
	advectparticle << <pblocknum, threadnum >> >(mParPos, mParVel, parNumNow,
		mwaterux, mwateruy, mwateruz, hparam.dt, parflag, velmode);
#else

	advectparticle_RK2 << <pblocknum, threadnum >> >(mParPos, mParVel, parNumNow,
		waterux, wateruy, wateruz, hparam.dt, parflag, velmode);
#endif
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}

void cspray::copyParticle2GL()
{
	float* renderPos, *rendercolor;
	cudaGLMapBufferObject((void**)&renderPos, vboParPos);
	cudaGLMapBufferObject((void**)&rendercolor, vboParColor);

	printf("rendering, parnumnow=%d\n", parNumNow);
	if (simmode == SIMULATION_SOLIDCOUPLING)
		copyParticle2GL_vel_k << <pblocknum, threadnum >> >(mParPos, mParVel, parmass, parflag, parNumNow, renderPos, rendercolor);
	else
		copyParticle2GL_phi << <pblocknum, threadnum >> >(mParPos, parflag, parmass, parTemperature, parNumNow, renderPos, rendercolor,
		lsmerge, phigrax, phigray, phigraz, renderpartiletype, temperatureMax_render, temperatureMin_render);

	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	cudaGLUnmapBufferObject(vboParPos);
	cudaGLUnmapBufferObject(vboParColor);
}

//参考：https://code.google.com/p/flip3d/
void cspray::correctpos()
{
	correctparticlepos << <pblocknum, threadnum >> >(tmpParPos, mParPos, parmass, parflag, parNumNow, gridstart, gridend, correctionspring, correctionradius,
		pEmptyPos, pEmptyRadius, pEmptyNum);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	float3 *temp = tmpParPos;
	tmpParPos = mParPos;
	mParPos = temp;
}

void cspray::mapvelp2g()
{
	mapvelp2g_k_fluidSolid << <gvblocknum, threadnum >> >(mParPos, mParVel, parmass, parflag, parNumNow, waterux, wateruy, wateruz, gridstart, gridend);
	//mapvelp2g_slow<<<gvblocknum, threadnum>>>(mParPos, mParVel, mParNum, mUx, mUy, mUz );
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	if (velmode == FLIP)
	{
		cudaMemcpy(waterux_old.data, waterux.data, sizeof(float)*hparam.gvnum.x, cudaMemcpyDeviceToDevice);
		cudaMemcpy(wateruy_old.data, wateruy.data, sizeof(float)*hparam.gvnum.y, cudaMemcpyDeviceToDevice);
		cudaMemcpy(wateruz_old.data, wateruz.data, sizeof(float)*hparam.gvnum.z, cudaMemcpyDeviceToDevice);
	}
}

inline void swappointer(farray &a, farray &b)
{
	float* temp = a.data;
	a.data = b.data;
	b.data = temp;
}

void cspray::mapvelg2p()
{
	if (velmode == FLIP)	//in CIP mode, vel of partciles will be updated in advect part.
	{		//注意，还要改一下速度的改变方式，是delta_v而不是v
		computeDeltaU << <gvblocknum, threadnum >> >(waterux, wateruy, wateruz, waterux_old, wateruy_old, wateruz_old);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed");

		mapvelg2p_flip << <pblocknum, threadnum >> >(mParPos, mParVel, parflag, parNumNow, waterux_old, wateruy_old, wateruz_old);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed");
	}
}

void cspray::addexternalforce()
{
	// 	if( mscene==SCENE_BUBBLE || mscene==SCENE_MULTIBUBBLE )
	// 		addgravityforce_k<<<pblocknum, threadnum>>>(mParVelFLIP, parflag, parTemperature, parNumNow, hparam.dt, Temperature0 );
	// 	else
	addgravityforce_k << <pblocknum, threadnum >> >(mParVel, parflag, parNumNow, hparam.dt);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	//addbuoyancyforce_k<<<pblocknum, threadnum>>>( buoyantHeight, mParPos,mParVel, parflag, parNumNow, hparam.dt );

	addbuoyancyforce_vel << <pblocknum, threadnum >> >(bubbleMaxVel, mParPos, mParVel, parflag, parNumNow, hparam.dt, buoyanceRateAir, buoyanceRateSolo);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	// 	buoyancyForSolid<<<pblocknum, threadnum>>>( mParPos, mParVel, parflag, parNumNow, gridstart, gridend, SolidBuoyanceParam );
	// 	cudaThreadSynchronize();
}

void cspray::initparticle_solidCoupling()
{
	if (parNumNow <= 0)
		return;

	float3* hparpos = new float3[parNumNow];//粒子位置
	float3* hparvel = new float3[parNumNow];//粒子速度
	float* hparmass = new float[parNumNow];//粒子质量
	char* hparflag = new char[parNumNow];	//粒子标记
	float x, y, z;

	int i = 0, ParNumPerLevel = 0;
	for (float z = hparam.cellsize.x + hparam.samplespace; z<0.8f * NZ*hparam.cellsize.x && i + ParNumPerLevel<initfluidparticle; z += hparam.samplespace)
	{
		// 		for( float y = hparam.cellsize.x+hparam.samplespace; y<hparam.cellsize.x*(0.7*NY-1)-0.5f*hparam.samplespace && i<initfluidparticle; y+=hparam.samplespace )
		// 			for( float x = hparam.cellsize.x+hparam.samplespace; x<hparam.cellsize.x*(0.7*NX-1)-0.5f*hparam.samplespace && i<initfluidparticle; x+=hparam.samplespace )
		// 			{
		for (float y = hparam.cellsize.x + hparam.samplespace; y<hparam.cellsize.x*(NY - 1) - 0.5f*hparam.samplespace && i<initfluidparticle; y += hparam.samplespace)
		for (float x = hparam.cellsize.x + hparam.samplespace; x<hparam.cellsize.x*(NX - 1) - 0.5f*hparam.samplespace && i<initfluidparticle; x += hparam.samplespace)
		{
			hparpos[i] = make_float3(x, y, z);
			hparvel[i] = make_float3(0.0f);
			hparmass[i] = hparam.m0;
			hparflag[i] = TYPEFLUID;
			++i;
		}
		if (ParNumPerLevel == 0) ParNumPerLevel = i;
	}

	float scale = 50;
	if (mscene == SCENE_FREEZING || mscene == SCENE_MELTINGPOUR) scale = 80;
	if (mscene == SCENE_INTERACTION) scale = 60;
	if (mscene == SCENE_MELTANDBOIL_HIGHRES || mscene == SCENE_INTERACTION_HIGHRES) scale = 100;

	if (m_bSolid)
	{
		for (int j = 0; j<nInitSolPoint; j++)
		{
			x = float(SolpointPos[j][0]), y = float(SolpointPos[j][1]), z = float(SolpointPos[j][2]);
			hparpos[i] = hparam.samplespace*make_float3(x, y, z)*scale + solidInitPos;
			hparvel[i] = make_float3(0.0f);		//	
			hparmass[i] = hparam.m0*0.8f;
			hparflag[i] = TYPESOLID;	//类型是固体

			++i;
		}
	}
	parNumNow = i;

	cudaMemcpy(mParPos, hparpos, sizeof(float3)*parNumNow, cudaMemcpyHostToDevice);
	cudaMemcpy(mParVel, hparvel, sizeof(float3)*parNumNow, cudaMemcpyHostToDevice);
	cudaMemcpy(parmass, hparmass, sizeof(float)*parNumNow, cudaMemcpyHostToDevice);
	cudaMemcpy(parflag, hparflag, sizeof(char)*parNumNow, cudaMemcpyHostToDevice);

	delete[] hparpos;
	delete[] hparvel;
	delete[] hparmass;
	delete[] hparflag;
}

void cspray::markgrid()
{
	markair << <gsblocknum, threadnum >> >(mmark);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	//markfluid<<<pblocknum, threadnum>>>( mmark, mParPos, parflag, parNumNow );
	//todo: 这里可能有问题！！！
	//markfluid_GY<<<pblocknum, threadnum>>>( mmark, mParPos, parflag, parNumNow );
	markfluid << <pblocknum, threadnum >> >(mmark, mParPos, parflag, parNumNow);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	if (bCouplingSphere)
	{
		markSolid_sphere << <gsblocknum, threadnum >> >(solidInitPos, sphereradius, mmark);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed");
	}

	//todo: 这里可能有问题！！！
	markBoundaryCell << <gsblocknum, threadnum >> >(mmark);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}

void cspray::flipMark_sphere()
{
	flipAirVacuum << <gsblocknum, threadnum >> >(mmark);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}

void cspray::markgrid_bubble()
{
	int fluidParCntPerGridThres = 10;
	markfluid_dense << <pblocknum, threadnum >> >(mmark, parmass, parflag, parNumNow, gridstart, gridend, fluidParCntPerGridThres);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
	markBoundaryCell << <gsblocknum, threadnum >> >(mmark);
}

void cspray::smokemarkgrid()
{
	markforsmoke << <gsblocknum, threadnum >> >(mmark, spraydense);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	if (bCouplingSphere)
	{
		markSolid_sphere << <gsblocknum, threadnum >> >(solidInitPos, sphereradius, mmark);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed");
	}

	markBoundaryCell << <gsblocknum, threadnum >> >(mmark);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}

void cspray::PrintMemInfo()
{
	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("freeMem=%fM,totalMem=%fM\n", freeMem / 1024.0f / 1024.0f, totalMem / 1024.0f / 1024.0f);
}

void cspray::setGridColor()
{
	float* color;
	cudaGLMapBufferObject((void**)&color, vboGridcolor);

	//printf( "%d\n", (int)colormode);
	setgridcolor_k << <gsblocknum, threadnum >> >(
		color, colormode, mpress, waterux_old, wateruy_old, wateruz_old, mDiv, phiair, mmark, lsmerge, Tp, surfacetensionsigma, temperatureMax_render, temperatureMin_render);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	cudaGLUnmapBufferObject(vboGridcolor);
}

void cspray::sweepPhi(farray phi, char typeflag)
{
	initphi << <gsblocknum, threadnum >> >(phi, mmark, typeflag);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	for (int it = 0; it<3; it++)
	{
		sweepphibytype << <gsblocknum, threadnum >> >(phi, mmark, typeflag);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed");
	}
}

void cspray::sweepU(farray ux, farray uy, farray uz, farray phi, charray mark, char typeflag)
{
	for (int it = 0; it<2; it++)
	{
		//sweepu<<<gvblocknum, threadnum>>>( tmpux,tmpuy,tmpuz, ux, uy, uz, phi, mark );
		sweepu_k_bubble << <gvblocknum, threadnum >> >(tmpux, tmpuy, tmpuz, ux, uy, uz, phi, mark, typeflag);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed");
		swappointer(tmpux, ux);
		swappointer(tmpuy, uy);
		swappointer(tmpuz, uz);
	}
}

void cspray::setSmokeBoundaryU(farray ux, farray uy, farray uz)
{
	setSmokeBoundaryU_k << <gvblocknum, threadnum >> >(ux, uy, uz, mmark);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}

void cspray::setWaterBoundaryU(farray ux, farray uy, farray uz)
{
	setWaterBoundaryU_k << <gvblocknum, threadnum >> >(ux, uy, uz, mmark);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}

extern void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

void cspray::hashAndSortParticles()
{
	calcHashD << <pblocknum, threadnum >> >(gridHash, gridIndex, mParPos, parNumNow);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	sortParticles(gridHash, gridIndex, parNumNow);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	//需要另一组数据来缓存
	swapParticlePointers();

	uint smemSize = sizeof(uint)*(threadnum + 1);
	(cudaMemset(gridstart, CELL_UNDEF, hparam.gnum*sizeof(uint)));
	reorderDataAndFindCellStartD << <pblocknum, threadnum, smemSize >> >(
		gridstart, gridend,
		mParPos, mParVel, parflag, parmass, parTemperature, parLHeat, parsolubility, pargascontain,
		gridHash, gridIndex,
		tmpParPos, tmpParVelFLIP, tmpparflag, tmpparmass, tmpparTemperature, tmpparHeat, tempsolubility, tempgascontain,
		parNumNow);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}

void cspray::hashAndSortParticles_MC()
{
	calcHashD_MC << <pblocknum, threadnum >> >(gridHash, gridIndex, mParPos, parNumNow);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	sortParticles(gridHash, gridIndex, parNumNow);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	//需要另一组数据来缓存
	swapParticlePointers();

	uint smemSize = sizeof(uint)*(threadnum + 1);
	(cudaMemset(gridstart, CELL_UNDEF, NXMC*NYMC*NZMC*sizeof(uint)));
	reorderDataAndFindCellStartD << <pblocknum, threadnum, smemSize >> >(
		gridstart, gridend,
		mParPos, mParVel, parflag, parmass, parTemperature, parLHeat, parsolubility, pargascontain,
		gridHash, gridIndex,
		tmpParPos, tmpParVelFLIP, tmpparflag, tmpparmass, tmpparTemperature, tmpparHeat, tempsolubility, tempgascontain,
		parNumNow);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}

void cspray::smokeadvection()
{
	//advect concertration by the div-free velocity field.
	advectscaler << <gsblocknum, threadnum >> >(tmpspraydense, spraydense, msprayux, msprayuy, msprayuz, densedissipation, wind);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	//advect velocity field by itself.
	advectux << <gvblocknum, threadnum >> >(mtmpsprayux, msprayux, msprayuy, msprayuz, velocitydissipation, wind);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
	advectuy << <gvblocknum, threadnum >> >(mtmpsprayuy, msprayux, msprayuy, msprayuz, velocitydissipation, wind);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
	advectuz << <gvblocknum, threadnum >> >(mtmpsprayuz, msprayux, msprayuy, msprayuz, velocitydissipation, wind);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	swappointer(msprayux, mtmpsprayux);
	swappointer(msprayuy, mtmpsprayuy);
	swappointer(msprayuz, mtmpsprayuz);
	swappointer(tmpspraydense, spraydense);
}

void cspray::smokesetvel()
{
	setsmokedense << <gsblocknum, threadnum >> > (spraydense);
	cudaThreadSynchronize();
	setsmokevel << <gvblocknum, threadnum >> >(msprayuz, spraydense);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}

void cspray::copyDensity2GL()
{
	cudaGraphicsMapResources(1, &densTex3D_cuda, 0);
	cudaArray *cudaarray;
	cudaGraphicsSubResourceGetMappedArray(&cudaarray, densTex3D_cuda, 0, 0);
	writedens2surface(cudaarray, gsblocknum, threadnum, tmpspraydense);

	cudaGraphicsUnmapResources(1, &densTex3D_cuda, 0);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}

void cspray::initMC()
{
	printf("Before MC: "), PrintMemInfo();

	uint* d_numVertsTable = 0;
	uint* d_edgeTable = 0;
	uint* d_triTable = 0;

	allocateTextures(&d_edgeTable, &d_triTable, &d_numVertsTable);

	int rate = 2 * 2 * 2;
	numVoxels = hparam.gnum * rate;		//NOTICE: *8 for double resolution
	maxVerts = 4000000;// hparam.gnum*1*rate;
	maxTriangles = 8000000;//hparam.gnum*2*rate;

	centertmp = make_float3(0);

	// allocate device memory
	unsigned int memSize = sizeof(uint)* numVoxels;
	checkCudaErrors(cudaMalloc((void**)&d_voxelVerts, memSize));
	checkCudaErrors(cudaMalloc((void**)&d_voxelVertsScan, memSize));
	checkCudaErrors(cudaMalloc((void**)&d_voxelOccupied, memSize));
	checkCudaErrors(cudaMalloc((void**)&d_voxelOccupiedScan, memSize));
	checkCudaErrors(cudaMalloc((void**)&d_compVoxelArray, memSize));

	MCedgeNum = 3 * (NX + 1)*(NY + 1)*(NZ + 1) * rate;			//NOTICE: *8 for double resolution
	cudaMalloc((void**)&MCedgemark, MCedgeNum*sizeof(uint));
	cudaMalloc((void**)&MCedgemarkScan, MCedgeNum*sizeof(uint));

	initMCtriangles(maxVerts, maxTriangles);
	//for smoothing the triangles in 3d mesh created by MC
	cudaMalloc((void**)&smoothdisplacement, maxVerts*sizeof(float3));
	cudaMalloc((void**)&smoothweight, maxVerts*sizeof(int));
	cudaMemset(smoothdisplacement, 0, maxVerts*sizeof(float3));		//necessary!!!
	cudaMemset(smoothweight, 0, maxVerts*sizeof(int));		//necessary!!!

	maxsolidvert = 300000, maxsolidtri = 600000;
	cudaMalloc((void**)&solidvertex, maxsolidvert*sizeof(float3));
	cudaMalloc((void**)&solidnormal, maxsolidvert*sizeof(float3));
	cudaMalloc((void**)&solidindices, maxsolidtri * 3 * sizeof(uint));

	//for marching cube.
	cudaMalloc((void**)&waterdensMC.data, sizeof(float)*(NX + 1)*(NY + 1)*(NZ + 1)*rate);
	waterdensMC.setdim(NX + 1, NY + 1, NZ + 1);

	printf("After MC: "), PrintMemInfo();
}

extern void ThrustScanWrapper(unsigned int* output, unsigned int* input, unsigned int numElements);

void cspray::preMC()
{
	NXMC = NX * 2, NYMC = NY * 2, NZMC = NZ * 2;
	copyNXNYNZtoGPU_MC(NXMC, NYMC, NZMC);

	hashAndSortParticles_MC();
}

void cspray::runMC_fluid()
{
	//将TYPEAIR, TYPEAIRSOLO和TYPEFLUID粒子用MC生成网格
	m_bLiquidAndGas = true;
	runMC_smooth("water", TYPEFLUID);
	m_bLiquidAndGas = false;
}

void cspray::runMC_solid()
{
	runMC_smooth("solid", TYPESOLID);
}

void cspray::runMC_gas()
{
	//将TYPEAIR, TYPEAIRSOLO粒子用MC生成网格
	m_bGas = true;
	runMC_smooth("gas", TYPEAIR);
	m_bGas = false;
}

void cspray::runMC_interaction()
{
	if (mframe>0 && !bRunMCSolid)//just output, they are updated in solidmotion function.
	{
		calNormals(solidnormal, solidvertex, solidvertexnum, solidindices, solidindicesnum);

		//output
		if (boutputpovray && mframe%outputframeDelta == 0)
			outputPovRaywater(mframe / outputframeDelta, solidvertex, solidnormal, solidvertexnum, solidindices, solidindicesnum, "solid");
	}
	else
		runMC_smooth("solid", TYPESOLID);
}

void cspray::runMC_smooth(const char* objectname, char MCParType)
{
	//1. gen the density field
	int blocknum = (int)ceil(((float)(NXMC + 1)*(NYMC + 1)*(NZMC + 1)) / threadnum);
	waterdensMC.setdim(NXMC + 1, NYMC + 1, NZMC + 1);

	//todo: gridstart and gridend.

	if (m_bLiquidAndGas)	// MC both liquid and gas particle.
		genWaterDensfield_liquidAndGas << <blocknum, threadnum >> >(waterdensMC, mParPos, parflag, gridstart, gridend, fMCDensity);
	else if (m_bGas)	// MC both liquid and gas particle.
		genWaterDensfield_Gas << <blocknum, threadnum >> >(waterdensMC, mParPos, parflag, gridstart, gridend, fMCDensity, mscene);
	else	if (m_DistanceFuncMC == 0)
	{
		genWaterDensfield_GY << <blocknum, threadnum >> >(waterdensMC, mParPos, parflag, gridstart, gridend, fMCDensity, MCParType, centertmp);
		//printf("dis_GY\n");
	}
	else
	{
		genWaterDensfield2 << <blocknum, threadnum >> >(waterdensMC, mParPos, parflag, gridstart, gridend, fMCDensity, MCParType);
		//printf("dis2\n");
	}

	//test
	// 	float3 pos = make_float3(0.25f, 0.25f, 0.25f + (mframe%100) * 0.001f );
	// 	genSphereDensfield<<<blocknum, threadnum>>>(waterdensMC, pos, 0.07f );

	//2. calculate number of vertices need per voxel
	blocknum = (int)ceil(((float)(NXMC)*(NYMC)*(NZMC)) / threadnum);
	numVoxels = NXMC*NYMC*NZMC;
	classifyVoxel << <blocknum, threadnum >> >(d_voxelVerts, d_voxelOccupied, waterdensMC, 0);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	//3. scan voxel occupied array
	ThrustScanWrapper(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

	// read back values to calculate total number of non-empty voxels
	// since we are using an exclusive scan, the total is the last value of
	// the scan result plus the last value in the input array
	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *)&lastElement,
			(void *)(d_voxelOccupied + numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
			(void *)(d_voxelOccupiedScan + numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		activeVoxels = lastElement + lastScanElement;
	}

	if (activeVoxels == 0) {
		// return if there are no full voxels
		totalVerts = 0;

		if (boutputpovray && mframe%outputframeDelta == 0)
			outputPovRaywater(mframe / outputframeDelta, NULL, NULL, 0, NULL, 0, objectname);
		return;
	}

	//4. compact voxel index array
	compactVoxels << <blocknum, threadnum >> >(d_compVoxelArray, d_voxelOccupied,
		d_voxelOccupiedScan, numVoxels);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	//5. scan voxel totalTriagles count array
	ThrustScanWrapper(d_voxelVertsScan, d_voxelVerts, numVoxels);
	// readback total number of totalTriagles
	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *)&lastElement,
			(void *)(d_voxelVerts + numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
			(void *)(d_voxelVertsScan + numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		totalIndices = (lastElement + lastScanElement);
		//		printf("indices number = %u\n", totalIndices);

		if (totalIndices>maxTriangles * 3)
		{
			printf("MC totalIndices exceeds, ERROR!!\n");
			mpause = true;
			return;
		}
	}

	//6. generate triangles, writing to vertex buffers
	{
		size_t num_bytes;
		float3 *d_pos, *d_normal;
		uint *d_indices;		//todo: gl memory
		checkCudaErrors(cudaGraphicsMapResources(1, &res_posvbo, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_pos, &num_bytes, res_posvbo));
		checkCudaErrors(cudaGraphicsMapResources(1, &res_normvbo, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_normal, &num_bytes, res_normvbo));
		checkCudaErrors(cudaGraphicsMapResources(1, &res_indicesvbo, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_indices, &num_bytes, res_indicesvbo));

		dim3 grid2((int)ceil(activeVoxels / (float)NTHREADS), 1, 1);
		while (grid2.x > 65535) {
			grid2.x /= 2;
			grid2.y *= 2;
		}

		//7. begin：为了包含三角形的索引关系，需要特殊处理
		{
			cudaMemset(MCedgemark, 0, MCedgeNum*sizeof(uint));

			markActiveEdge_MC << <grid2, NTHREADS >> >(MCedgemark, d_compVoxelArray, waterdensMC, 0, activeVoxels);
			cudaThreadSynchronize();
			getLastCudaError("Kernel execution failed");
			// scan voxel vertex count array

			ThrustScanWrapper(MCedgemarkScan, MCedgemark, MCedgeNum);

			// readback total number of vertices
			{
				uint lastElement, lastScanElement;
				checkCudaErrors(cudaMemcpy((void *)&lastElement,
					(void *)(MCedgemark + MCedgeNum - 1),
					sizeof(uint), cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
					(void *)(MCedgemarkScan + MCedgeNum - 1),
					sizeof(uint), cudaMemcpyDeviceToHost));
				totalVerts = lastElement + lastScanElement;
			}
			if (totalVerts>maxVerts)
			{
				printf("MC total verts exceed, ERROR!!\n");
				mpause = true;
				return;
			}
			//	printf("totalVerts number = %u\n", totalVerts);

			generateTriangles_indices << <grid2, NTHREADS >> >(d_pos, d_indices, d_compVoxelArray, waterdensMC, 0,
				activeVoxels, maxVerts, MCedgemarkScan, d_voxelVertsScan);
			cudaThreadSynchronize();
			getLastCudaError("Kernel execution failed");
		}

		//8. smooth and calculate normals of vertices.
		if (m_bSmoothMC)
			smoothMesh(d_pos, totalVerts, d_indices, totalIndices / 3);
		calNormals(d_normal, d_pos, totalVerts, d_indices, totalIndices);

		//output
		if (boutputpovray && mframe%outputframeDelta == 0)
			outputPovRaywater(mframe / outputframeDelta, d_pos, d_normal, totalVerts, d_indices, totalIndices, objectname);

		if ((mscene == SCENE_INTERACTION || mscene == SCENE_INTERACTION_HIGHRES || mscene == SCENE_MELTANDBOIL || mscene == SCENE_MELTANDBOIL_HIGHRES) && objectname == "solid")
		{
			if (totalVerts>maxsolidvert || totalIndices>maxsolidtri * 3)		//mem is not enough, error.
			{
				printf("MC for solid: vert and triangle are too many!!! ERROR!\n");
				mpause = true;
				return;
			}
			cudaMemcpy(solidvertex, d_pos, totalVerts*sizeof(float3), cudaMemcpyDeviceToDevice);
			cudaMemcpy(solidindices, d_indices, totalIndices*sizeof(uint), cudaMemcpyDeviceToDevice);
			solidvertexnum = totalVerts;
			solidindicesnum = totalIndices;
		}

		printf(" vert num=%d, indices=%d\n", totalVerts, totalIndices);

		checkCudaErrors(cudaGraphicsUnmapResources(1, &res_normvbo, 0));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &res_posvbo, 0));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &res_indicesvbo, 0));
	}
}

void cspray::calNormals(float3 *dnormals, float3 *dpos, int vertexnum, uint *dindices, int indicesnum)
{
	//set to 0
	cudaMemset(dnormals, 0, vertexnum * 3 * sizeof(float));

	//calculate face normal
	int faceblocknum = max(1, (int)ceil(indicesnum / 3.0f / threadnum));
	calnormal_k << <faceblocknum, threadnum >> >(dpos, dnormals, vertexnum, dindices, indicesnum);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	//calculate vertex normal.
	int vertexblocknum = max(1, (int)ceil(((float)vertexnum) / threadnum));
	normalizeTriangleNor_k << <vertexblocknum, threadnum >> >(dnormals, vertexnum);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}

bool verifyfloat3(float3 &a)
{
	if (!(a.x>0 || a.x <1))
		return false;
	if (!(a.y>0 || a.y <1))
		return false;
	if (!(a.z>0 || a.z <1))
		return false;
	return true;
}

void cspray::runMC_flat(char MCParType)
{
	//gen the density field
	int blocknum = (int)ceil(((float)(NX + 1)*(NY + 1)*(NZ + 1)) / threadnum);

	//	genWaterDensfield<<<blocknum, threadnum>>>( waterdensMC, mParPos, parflag, gridstart, gridend, fMCDensity);
	//genSphereDensfield<<<blocknum, threadnum>>>( waterdensMC );
	genWaterDensfield_GY << <blocknum, threadnum >> >(waterdensMC, mParPos, parflag, gridstart, gridend, fMCDensity, MCParType, centertmp);

	// calculate number of vertices need per voxel
	classifyVoxel << <gsblocknum, threadnum >> >(d_voxelVerts, d_voxelOccupied, waterdensMC, 0);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	// scan voxel occupied array
	ThrustScanWrapper(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

	// read back values to calculate total number of non-empty voxels
	// since we are using an exclusive scan, the total is the last value of
	// the scan result plus the last value in the input array
	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *)&lastElement,
			(void *)(d_voxelOccupied + hparam.gnum - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
			(void *)(d_voxelOccupiedScan + hparam.gnum - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		activeVoxels = lastElement + lastScanElement;
	}

	if (activeVoxels == 0) {
		// return if there are no full voxels
		totalVerts = 0;
		return;
	}

	// compact voxel index array
	compactVoxels << <gsblocknum, threadnum >> >(d_compVoxelArray, d_voxelOccupied,
		d_voxelOccupiedScan, hparam.gnum);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	// scan voxel vertex count array
	ThrustScanWrapper(d_voxelVertsScan, d_voxelVerts, numVoxels);
	// readback total number of vertices
	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *)&lastElement,
			(void *)(d_voxelVerts + numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
			(void *)(d_voxelVertsScan + numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		totalVerts = lastElement + lastScanElement;
	}
	printf("totalVerts number = %u\n", totalVerts);

	// generate triangles, writing to vertex buffers
	{
		size_t num_bytes;
		float3 *d_pos, *d_normal;
		// DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_pos, posVbo));
		checkCudaErrors(cudaGraphicsMapResources(1, &res_posvbo, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_pos, &num_bytes, res_posvbo));

		// DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&d_normal, normalVbo));
		checkCudaErrors(cudaGraphicsMapResources(1, &res_normvbo, 0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_normal, &num_bytes, res_normvbo));

		dim3 grid2((int)ceil(activeVoxels / (float)NTHREADS), 1, 1);
		while (grid2.x > 65535) {
			grid2.x /= 2;
			grid2.y *= 2;
		}
		generateTriangles2 << <grid2, NTHREADS >> >(d_pos, d_normal, d_compVoxelArray, d_voxelVertsScan, waterdensMC, 0,
			activeVoxels, maxVerts);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed");
		// DEPRECATED: 		checkCudaErrors(cudaGLUnmapBufferObject(normalVbo));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &res_normvbo, 0));
		// DEPRECATED: 		checkCudaErrors(cudaGLUnmapBufferObject(posVbo));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &res_posvbo, 0));
	}
}

int cspray::getblocknum(int n)
{
	return (int)ceil(((float)n) / threadnum);
}

float cspray::product(farray a, farray b, int n)
{
	static float *dsum = NULL;
	if (!dsum)
		cudaMalloc((void**)&dsum, sizeof(float)*gsblocknum);
	static float *hsum = new float[gsblocknum];
	int sharememsize = threadnum*sizeof(float);

	arrayproduct_k << <gsblocknum, threadnum, sharememsize >> >(dsum, a.data, b.data, n);
	cudaThreadSynchronize();
	cudaMemcpy(hsum, dsum, sizeof(float)*gsblocknum, cudaMemcpyDeviceToHost);
	float res = 0;
	for (int i = 0; i<gsblocknum; i++)
		res += hsum[i];//, printf( "debug: hsum = %f\n", hsum[i]);
	return res;
}

bool verifyfloat(float a)
{
	if (!(a>0 || a<1))
		return false;
	return true;
}

bool cspray::solver_cg(charray A, farray x, farray b, int n)
{
	//todo: build precondition

	float a, zp, error2, eps = ((float)(1e-2))*n, alpha, beta;
	// z = applyA(x)
	computeAx << <gsblocknum, threadnum >> >(z, A, x, n);
	// r = b-Ax = b-1.0*z
	pcg_op << <gsblocknum, threadnum >> >(A, r, b, z, -1.0, n);

	error2 = product(r, r, n);
	if (error2<eps)
		return true;
	//error2 = product( r, r, n );	//error2 = r * r

	//todo: applyPreconditioner
	//buildprecondition_pcg<<<gsblocknum,threadnum>>>( pre, A, z,r,n);
	cudaMemcpy(z.data, r.data, sizeof(float)*n, cudaMemcpyDeviceToDevice);	//z=r;

	cudaMemcpy(p.data, z.data, sizeof(float)*n, cudaMemcpyDeviceToDevice);	//p=z;

	a = product(z, r, n);	//a = z*r;

	int k = 0;
	for (; k<n; k++)
	{
		//for debug.
		if (k >= 800 && k % 100 == 0)
		{
			if (!verifyfloat(a) || !verifyfloat(alpha) || !verifyfloat(error2))
				printf("there is some illegal float number in PCG solver!!!\n");
			printf("pcg iteration times: %d\n", k);
		}

		//z=A*p
		computeAx << <gsblocknum, threadnum >> >(z, A, p, n);

		zp = product(z, p, n);	//zp = z*p
		if (zp == 0)
			return true;

		alpha = a / zp;		//alpha = a/(z . p) = z*r/(z*p)

		// x = x + alpha*p
		pcg_op << <gsblocknum, threadnum >> >(A, x, x, p, alpha, n);
		// r = r - alpha*z;
		pcg_op << <gsblocknum, threadnum >> >(A, r, r, z, -alpha, n);

		//error2 = r * r
		error2 = product(r, r, n);
		if (error2<eps)
			break;

		//todo: applyPreconditioner
		cudaMemcpy(z.data, r.data, sizeof(float)*n, cudaMemcpyDeviceToDevice);	//z=r;
		//buildprecondition_pcg<<<gsblocknum,threadnum>>>( pre, A, z,r,n);

		//a2 = z*r;
		float a2 = product(z, r, n);

		beta = a2 / a;                     // beta = a2 / a
		// p = z + beta*p
		pcg_op << <gsblocknum, threadnum >> >(A, p, z, p, beta, n);

		a = a2;
	}
	// 	if( mtime )
	// 		printf("CG interation: %d\n", k );

	return true;
}

void cspray::solver_Jacobi(charray A, farray x, farray b, int itertime)
{
	cudaMemset(temppress.data, 0, sizeof(float)*hparam.gnum);
	for (int i = 0; i<itertime / 2; i++)
	{
		JacobiIter << <gsblocknum, threadnum >> >(temppress, x, b, A);
		cudaThreadSynchronize();

		JacobiIter << <gsblocknum, threadnum >> >(x, temppress, b, A);
		cudaThreadSynchronize();
	}
}

void cspray::swapParticlePointers()
{
	float3* temp;
	temp = mParPos, mParPos = tmpParPos, tmpParPos = temp;
	temp = mParVel, mParVel = tmpParVelFLIP, tmpParVelFLIP = temp;
	float* temp2;
	temp2 = tmpparmass, tmpparmass = parmass, parmass = temp2;
	temp2 = tmpparTemperature, tmpparTemperature = parTemperature, parTemperature = temp2;
	temp2 = tmpparHeat, tmpparHeat = parLHeat, parLHeat = temp2;
	temp2 = tempsolubility, tempsolubility = parsolubility, parsolubility = temp2;
	temp2 = tempgascontain, tempgascontain = pargascontain, pargascontain = temp2;
	char* temp3;
	temp3 = tmpparflag, tmpparflag = parflag, parflag = temp3;
}

void cspray::checkdensesum()
{
	static float *hdense = new float[hparam.gnum];
	cudaMemcpy(hdense, spraydense.data, sizeof(float)*hparam.gnum, cudaMemcpyDeviceToHost);
	float sum = 0;
	for (int i = 0; i<hparam.gnum; i++)
		sum += hdense[i];
	printf("dense sum = %f\n", sum);

	static float dense0 = sum;
	printf("dense0 = %f\n", dense0);
	for (int i = 0; i<hparam.gnum; i++)
		hdense[i] *= dense0 / sum;
	cudaMemcpy(spraydense.data, hdense, sizeof(float)*hparam.gnum, cudaMemcpyHostToDevice);
}

void cspray::checkparticlevariables(float3* dvel)
{
	//debug if off.
#if 1
	static float3* hvel = new float3[parNumMax];
	cudaMemcpy(hvel, dvel, parNumNow*sizeof(float3), cudaMemcpyDeviceToHost);
	for (int i = 0; i<parNumNow; i++)
	{
		if (!(hvel[i].x>-1 || hvel[i].x<1))
		{
			printf("i=%d, particle velocity x=%f!!!!!\n", i, hvel[i].x);
			mpause = true;
			return;
		}
		if (!(hvel[i].y>-1 || hvel[i].y<1))
		{
			printf("i=%d, particle velocity y=%f!!!!!\n", i, hvel[i].y);
			mpause = true;
			return;
		}
		if (!(hvel[i].z>-1 || hvel[i].z<1))
		{
			printf("i=%d, particle velocity z=%f!!!!!\n", i, hvel[i].z);
			mpause = true;
			return;
		}
	}
#endif
}

void cspray::smoothMesh(float3 *dpos, int vertexnum, uint *indices, int trianglenum)
{
	int fblocknum = max(1, (int)ceil(((float)trianglenum) / threadnum));
	int vblocknum = max(1, (int)ceil(((float)vertexnum) / threadnum));
	float lambda = 0.5f;
	float mu = -0.53f;

	for (int i = 0; i<smoothIterTimes; i++)
	{
		smooth_computedisplacement << <fblocknum, threadnum >> >(smoothdisplacement, smoothweight, dpos, indices, trianglenum);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed");
		smooth_addDisplacement << <vblocknum, threadnum >> >(smoothdisplacement, smoothweight, dpos, vertexnum, lambda);
		cudaThreadSynchronize();
		smooth_addDisplacement << <vblocknum, threadnum >> >(smoothdisplacement, smoothweight, dpos, vertexnum, mu);
		cudaThreadSynchronize();
	}
}

//[GPU Gems]Fast Fluid Dynamics Simulation on the GPU
void cspray::smokediffuse()
{
	//diffuse dense field.
	float alpha, beta;
	alpha = hparam.cellsize.x*hparam.cellsize.x / hparam.dt / fDenseDiffuse;
	beta = 6 + alpha;
	for (int i = 0; i<nDiffuseIters / 2; i++)
	{
		diffuse_dense << <gsblocknum, threadnum >> >(tmpspraydense, spraydense, mmark, alpha, beta);
		cudaThreadSynchronize();
		diffuse_dense << <gsblocknum, threadnum >> >(spraydense, tmpspraydense, mmark, alpha, beta);
		cudaThreadSynchronize();
	}
}

void cspray::ComputeTriangleHashSize(myMesh &mesh)
{
	//1. 把场景中的小三角形hash起来
	int numFaces = mesh.m_nFaces;
	int nGridDim = (int)ceil(((float)numFaces) / threadnum);

	float* dMaxLength, *hHashSize;
	checkCudaErrors(cudaMalloc((void**)&dMaxLength, sizeof(float)*nGridDim));
	hHashSize = new float[nGridDim];

	createAABB_q << <nGridDim, threadnum >> >(mesh.m_dPoints,
		mesh.m_nPoints, mesh.m_dFaces, numFaces, dMaxLength, mesh.m_dHashPointsForFaces);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	//2. 计算三角形Hash网格的大小 
	cudaMemcpy(hHashSize, dMaxLength, sizeof(float)*nGridDim, cudaMemcpyDeviceToHost);
	float hashSize = hHashSize[0];
	for (int i = 1; i<nGridDim; i++)
	{
		hashSize = max(hHashSize[i], hashSize);
	}

	checkCudaErrors(cudaFree(dMaxLength));
	delete[] hHashSize;

	hashSize = hashSize / 2 + hparam.pradius;

	hparam.triHashSize = make_float3(hashSize);
	hparam.triHashRes.x = ceil((mesh.m_max.x - mesh.m_min.x) / hparam.triHashSize.x);
	hparam.triHashRes.y = ceil((mesh.m_max.y - mesh.m_min.y) / hparam.triHashSize.y);
	hparam.triHashRes.z = ceil((mesh.m_max.z - mesh.m_min.z) / hparam.triHashSize.z);
	hparam.triHashCells = (int)(hparam.triHashRes.x*hparam.triHashRes.y*hparam.triHashRes.z);

	copyparamtoGPU(hparam);

	//分配空间
	checkCudaErrors(cudaMalloc((void**)&mesh.m_dTriCellStart, sizeof(uint)*hparam.triHashCells));
	checkCudaErrors(cudaMalloc((void**)&mesh.m_dTriCellEnd, sizeof(uint)*hparam.triHashCells));
}

void cspray::hashTriangle_radix_q()
{
	int numFaces = mmesh.m_nFaces;
	int nGridDim = (int)ceil(((float)numFaces) / threadnum);

	calcHash_radix_q << < nGridDim, threadnum >> >((uint2*)mmesh.m_dTriHash_radix[0], mmesh.m_dHashPointsForFaces, numFaces, mmesh.m_min, mmesh.m_max);

	getLastCudaError("Kernel execution failed");
	cudaThreadSynchronize();
}

void cspray::sortTriangles_q(uint numParticles)
{
	RadixSort((KeyValuePair *)mmesh.m_dTriHash_radix[0], (KeyValuePair *)mmesh.m_dTriHash_radix[1], numParticles, 32);
	getLastCudaError("Kernel execution failed");
	cudaThreadSynchronize();
}


void cspray::reorderTriangle_radix_q()
{
	int numFaces = mmesh.m_nFaces;

	checkCudaErrors(cudaMemset(mmesh.m_dTriCellStart, CELL_UNDEF, hparam.triHashCells * sizeof(uint)));
	int nGridDim = (int)ceil(((float)numFaces) / threadnum);
	uint smemSize = sizeof(uint)*(threadnum + 1);


	reorderDataAndFindCellStart_radix_q << <nGridDim, threadnum, smemSize >> >(
		mmesh.m_dTriCellStart, mmesh.m_dTriCellEnd,
		mmesh.m_dFacesSorted,
		(uint2*)mmesh.m_dTriHash_radix[0],
		mmesh.m_dFaces,
		numFaces);


	getLastCudaError("Kernel execution failed");
	cudaThreadSynchronize();

}
void cspray::updateNormal_q()
{

	int numFaces = mmesh.m_nFaces;
	int nGridDim = (int)ceil(((float)numFaces) / threadnum);

	calculateNormal << <nGridDim, threadnum >> >(mmesh.m_dPoints, mmesh.m_dFacesSorted, mmesh.m_dFaceNormals, mmesh.m_nFaces);

	getLastCudaError("Kernel execution failed");
	cudaThreadSynchronize();
}

void cspray::initscene_bubble()
{
	if (parNumNow <= 0)
		return;

	float3* hparpos = new float3[parNumNow];
	float3* hparvel = new float3[parNumNow];
	float* hparmass = new float[parNumNow];
	//	float* htemperature = new float[parNumNow];
	char* hparflag = new char[parNumNow];

	float3 bubblepos = make_float3(NX*0.5f, NY*0.5f, 8.5f) * hparam.cellsize.x;		//bottom
	float bubbleradius2 = hparam.cellsize.x * 0.6f;		//small bubble

	//1. 初始化流体部分
	int i = 0;
	for (float z = hparam.cellsize.x + hparam.samplespace; z<0.8f * NZ*hparam.cellsize.x && i<initfluidparticle; z += hparam.samplespace)
	{
		for (float y = hparam.cellsize.x + hparam.samplespace; y<hparam.cellsize.x*(NY - 1) - 0.5f*hparam.samplespace && i<initfluidparticle; y += hparam.samplespace)
		for (float x = hparam.cellsize.x + hparam.samplespace; x<hparam.cellsize.x*(NX - 1) - 0.5f*hparam.samplespace && i<initfluidparticle; x += hparam.samplespace)
		{
			hparpos[i] = make_float3(x, y, z);
			hparvel[i] = make_float3(0.0f);
			hparmass[i] = hparam.m0;

			if (length(hparpos[i] - bubblepos)<bubbleradius2)
				hparflag[i] = TYPEAIR;
			else
				hparflag[i] = TYPEFLUID;

			++i;
		}
	}
	parNumNow = i;
	printf("init fluid/air particle succeed, parnum=%d.\n", parNumNow);

	//2. 如果有固体的话，初始化固体部分
	if (m_bSolid)
	{
		for (int j = 0; j<nInitSolPoint; j++)
		{
			float x = float(SolpointPos[j][0]), y = float(SolpointPos[j][1]), z = float(SolpointPos[j][2]);
			hparpos[i + j] = hparam.samplespace*make_float3(x, y, z) * 50 + solidInitPos;
			//	printf( "%f,%f,%f\n", hparpos[i+j].x, hparpos[i+j].y, hparpos[i+j].z );

			hparvel[i + j] = make_float3(0.0f);		//	
			hparmass[i + j] = hparam.m0*0.8f;
			hparflag[i + j] = TYPESOLID;	//类型是固体6692963
		}
		parNumNow += nInitSolPoint;
		printf("init solid particle succeed, parnum=%d.\n", parNumNow);
	}

	cudaMemcpy(mParPos, hparpos, sizeof(float3)*parNumNow, cudaMemcpyHostToDevice);
	cudaMemcpy(mParVel, hparvel, sizeof(float3)*parNumNow, cudaMemcpyHostToDevice);
	cudaMemcpy(parmass, hparmass, sizeof(float)*parNumNow, cudaMemcpyHostToDevice);
	//	cudaMemcpy( parTemperature, htemperature, sizeof(float)*parNumNow, cudaMemcpyHostToDevice);
	cudaMemcpy(parflag, hparflag, sizeof(char)*parNumNow, cudaMemcpyHostToDevice);

	delete[] hparpos;
	delete[] hparvel;
	delete[] hparmass;
	//delete [] htemperature;
	delete[] hparflag;
}

void cspray::computeLevelset(float offset)
{
	//注意这个函数的mark和levelset是很关键的，得小心对待。
	genlevelset << <gsblocknum, threadnum >> >(lsfluid, lsair, mmark, mParPos, parflag, parmass, gridstart, gridend, fMCDensity, offset);
	getLastCudaError("Kernel execution failed");
	cudaThreadSynchronize();
}

inline bool verifycellidx(int i, int j, int k)
{
	if (i<0 || i>NX - 1 || j<0 || j>NY - 1 || k<0 || k>NZ - 1)
		return false;
	return true;
}

inline float sharp_kernel(float r2, float h)
{
	return fmax(h*h / fmax(r2, 0.0001f) - 1.0f, 0.0f);
}

void sumcell_host(float3 &usum, float &weight, float3 gpos, float3 *pos, float3 *vel, float *mass, uint *gridstart, uint  *gridend, int gidx)
{
	if (gridstart[gidx] == CELL_UNDEF)
		return;
	uint start = gridstart[gidx];
	uint end = gridend[gidx];
	float dis2, w, RE = 1.4;
	float scale = 64;
	for (uint p = start; p<end; ++p)
	{
		dis2 = dot(pos[p] * scale - gpos, pos[p] * scale - gpos);		//scale is necessary.
		w = mass[p] * sharp_kernel(dis2, RE);
		weight += w;
		usum += w*vel[p];
	}
}

void cspray::mapvelp2g_bubble_CPU()
{
	cudaMemcpy(hgridstart, gridstart, hparam.gnum*sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(hgridend, gridend, hparam.gnum*sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(hpos, mParPos, parNumNow*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(hvel, mParVel, parNumNow*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(hmass, parmass, parNumNow*sizeof(float), cudaMemcpyDeviceToHost);

	CTimer time;
	time.startTimer();

	int vnum = max((NX + 1)*NY*NZ, (NX)*(NY + 1)*NZ);
	vnum = max(vnum, NX*NY*(NZ + 1));
	float weight;
	float3 gpos, usum;
	int i, j, k;
	for (int idx = 0; idx<vnum; idx++)
	{
		//ux
		if (idx<hparam.gvnum.x)
		{
			weight = 0, usum = make_float3(0.0f);
			getijk(i, j, k, idx, NX + 1, NY, NZ);
			gpos.x = i, gpos.y = j + 0.5, gpos.z = k + 0.5;
			for (int di = -1; di <= 0; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++)
			if (verifycellidx(i + di, j + dj, k + dk))
				sumcell_host(usum, weight, gpos, hpos, hvel, hmass, hgridstart, hgridend, getidx(i + di, j + dj, k + dk));

			usum.x = (weight>0) ? (usum.x / weight) : 0.0f;
			hwaterux(i, j, k) = usum.x;
		}
		// uy
		if (idx<hparam.gvnum.y)
		{
			weight = 0, usum = make_float3(0.0f);
			getijk(i, j, k, idx, NX, NY + 1, NZ);
			gpos.x = i + 0.5, gpos.y = j, gpos.z = k + 0.5;
			for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 0; dj++) for (int dk = -1; dk <= 1; dk++)
			if (verifycellidx(i + di, j + dj, k + dk))
				sumcell_host(usum, weight, gpos, hpos, hvel, hmass, hgridstart, hgridend, getidx(i + di, j + dj, k + dk));
			usum.y = (weight>0) ? (usum.y / weight) : 0.0f;
			hwateruy(i, j, k) = usum.y;
		}
		// uz
		if (idx<hparam.gvnum.z)
		{
			weight = 0, usum = make_float3(0.0f);
			getijk(i, j, k, idx, NX, NY, NZ + 1);
			gpos.x = i + 0.5, gpos.y = j + 0.5, gpos.z = k;
			for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 0; dk++)
			if (verifycellidx(i + di, j + dj, k + dk))
				sumcell_host(usum, weight, gpos, hpos, hvel, hmass, hgridstart, hgridend, getidx(i + di, j + dj, k + dk));
			usum.z = (weight>0) ? (usum.z / weight) : 0.0f;
			hwateruz(i, j, k) = usum.z;
		}

	}
	printTime(m_bCPURun, "mapvelp2g_bubble_CPU", time);
}

void cspray::mapvelp2g_bubble()
{

	if (m_bCPURun)
		mapvelp2g_bubble_CPU();

	CTimer time;
	time.startTimer();
	mapvelp2g_k_fluidSolid << <gvblocknum, threadnum >> >(mParPos, mParVel, parmass, parflag, parNumNow, waterux, wateruy, wateruz, gridstart, gridend);
	cudaThreadSynchronize();
	printTime(m_bCPURun, "mapvelp2g_k_fluidSolid", time);

	mapvelp2g_k_air << <gvblocknum, threadnum >> >(mParPos, mParVel, parmass, parflag, parNumNow, airux, airuy, airuz, gridstart, gridend);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	if (velmode == FLIP)
	{
		cudaMemcpy(waterux_old.data, waterux.data, sizeof(float)*hparam.gvnum.x, cudaMemcpyDeviceToDevice);
		cudaMemcpy(wateruy_old.data, wateruy.data, sizeof(float)*hparam.gvnum.y, cudaMemcpyDeviceToDevice);
		cudaMemcpy(wateruz_old.data, wateruz.data, sizeof(float)*hparam.gvnum.z, cudaMemcpyDeviceToDevice);

		cudaMemcpy(airux_old.data, airux.data, sizeof(float)*hparam.gvnum.x, cudaMemcpyDeviceToDevice);
		cudaMemcpy(airuy_old.data, airuy.data, sizeof(float)*hparam.gvnum.y, cudaMemcpyDeviceToDevice);
		cudaMemcpy(airuz_old.data, airuz.data, sizeof(float)*hparam.gvnum.z, cudaMemcpyDeviceToDevice);
	}
}

void cspray::project_CG_bubble()
{
	CTimer time;
	time.startTimer();
	cudaMemset(mDiv.data, 0, sizeof(float)*hparam.gnum);
	cptdivergence_bubble2 << <gsblocknum, threadnum >> >(mDiv, waterux, wateruy, wateruz, airux, airuy, airuz, mmark, lsmerge);
	//cptdivergence<<<gsblocknum,threadnum>>>( mDiv, waterux, wateruy, wateruz, mmark );
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	//solve pressure
	cudaMemset(mpress.data, 0, sizeof(float)*hparam.gnum);
	solver_cg_bubble(mmark, mpress, mDiv, hparam.gnum);

	//compute divergence-free velocity.
	subGradPress << <gvblocknum, threadnum >> >(mpress, waterux, wateruy, wateruz);
	subGradPress << <gvblocknum, threadnum >> >(mpress, airux, airuy, airuz);
	// subGradPress_bubble<<<gvblocknum,threadnum>>>(mpress, waterux, wateruy, wateruz, surfacetension, lsmerge, mmark );
	//	subGradPress_bubble<<<gvblocknum,threadnum>>>(mpress, airux, airuy, airuz, surfacetension, lsmerge, mmark );
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}

bool cspray::solver_cg_bubble(charray A, farray x, farray b, int n)
{
	//todo: build precondition
	// 	checkGridFarray( b );
	// 	printf( "b\n" );

	float a, zp, error2, eps = ((float)(1e-2))*n, alpha, beta;
	// z = applyA(x)
	computeAx_bubble << <gsblocknum, threadnum >> >(z, A, x, n);
	// r = b-Ax = b-1.0*z
	pcg_op_bubble << <gsblocknum, threadnum >> >(A, r, b, z, -1.0, n);

	error2 = product(r, r, n);
	if (error2<eps)
		return true;
	//error2 = product( r, r, n );	//error2 = r * r

	//todo: applyPreconditioner
	//buildprecondition_pcg<<<gsblocknum,threadnum>>>( pre, A, z,r,n);
	cudaMemcpy(z.data, r.data, sizeof(float)*n, cudaMemcpyDeviceToDevice);	//z=r;

	cudaMemcpy(p.data, z.data, sizeof(float)*n, cudaMemcpyDeviceToDevice);	//p=z;

	a = product(z, r, n);	//a = z*r;

	int k = 0;
	for (; k<n; k++)
	{
		//for debug.
		if (k >= 800 && k % 100 == 0)
		{
			if (!verifyfloat(a) || !verifyfloat(alpha) || !verifyfloat(error2))
				printf("there is some illegal float number in PCG solver!!!\n");
			printf("pcg iteration times: %d\n", k);
			mpause = true;
			break;
		}

		//z=A*p
		computeAx_bubble << <gsblocknum, threadnum >> >(z, A, p, n);

		zp = product(z, p, n);	//zp = z*p
		if (zp == 0)
			return true;

		alpha = a / zp;		//alpha = a/(z . p) = z*r/(z*p)

		// x = x + alpha*p
		pcg_op_bubble << <gsblocknum, threadnum >> >(A, x, x, p, alpha, n);
		// r = r - alpha*z;
		pcg_op_bubble << <gsblocknum, threadnum >> >(A, r, r, z, -alpha, n);

		//error2 = r * r
		error2 = product(r, r, n);
		if (error2<eps)
			break;

		//todo: applyPreconditioner
		cudaMemcpy(z.data, r.data, sizeof(float)*n, cudaMemcpyDeviceToDevice);	//z=r;
		//buildprecondition_pcg<<<gsblocknum,threadnum>>>( pre, A, z,r,n);

		//a2 = z*r;
		float a2 = product(z, r, n);

		beta = a2 / a;                     // beta = a2 / a
		// p = z + beta*p
		pcg_op_bubble << <gsblocknum, threadnum >> >(A, p, z, p, beta, n);

		a = a2;
	}
	// 	if( mtime )
	// 		printf("CG interation: %d\n", k );

	return true;
}

void cspray::mapvelg2p_bubble()
{
	if (velmode == FLIP)	//in CIP mode, vel of partciles will be updated in advect part.
	{		//注意，还要改一下速度的改变方式，是delta_v而不是v
		computeDeltaU << <gvblocknum, threadnum >> >(waterux, wateruy, wateruz, waterux_old, wateruy_old, wateruz_old);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed");

		computeDeltaU << <gvblocknum, threadnum >> >(airux, airuy, airuz, airux_old, airuy_old, airuz_old);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed");

		mapvelg2p_flip_bubble << <pblocknum, threadnum >> >(mParPos, mParVel, parflag, parNumNow, waterux_old, wateruy_old, wateruz_old, airux_old, airuy_old, airuz_old);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed");
	}
}

inline float trilinear(farray u, float x, float y, float z, int w, int h, int d)
{
	x = fmax(0.0f, fmin(x, w));
	y = fmax(0.0f, fmin(y, h));
	z = fmax(0.0f, fmin(z, d));
	int i = fmin(x, w - 2);
	int j = fmin(y, h - 2);
	int k = fmin(z, d - 2);

	return (k + 1 - z)*((j + 1 - y)*((i + 1 - x)*u(i, j, k) + (x - i)*u(i + 1, j, k)) + (y - j)*((i + 1 - x)*u(i, j + 1, k) + (x - i)*u(i + 1, j + 1, k))) +
		(z - k)*((j + 1 - y)*((i + 1 - x)*u(i, j, k + 1) + (x - i)*u(i + 1, j, k + 1)) + (y - j)*((i + 1 - x)*u(i, j + 1, k + 1) + (x - i)*u(i + 1, j + 1, k + 1)));
}

float3 cspray::getParticleVelFromGrid(float3 pos, farray ux, farray uy, farray uz)
{
	float3 vel;
	float x = pos.x, y = pos.y, z = pos.z;
	x /= hparam.cellsize.x;
	y /= hparam.cellsize.y;
	z /= hparam.cellsize.z;

	//注意：ux,uy,uz的存储方式比较特殊(staggered grid)，三维线性插值也要比较小心
	vel.x = trilinear(ux, x, y - 0.5f, z - 0.5f, NX + 1, NY, NZ);
	vel.y = trilinear(uy, x - 0.5f, y, z - 0.5f, NX, NY + 1, NZ);
	vel.z = trilinear(uz, x - 0.5f, y - 0.5f, z, NX, NY, NZ + 1);
	return vel;
}

void cspray::advect_bubble_CPU()
{

	cudaMemcpy(hpos, mParPos, parNumNow*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(hvel, mParVel, parNumNow*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(hwaterux.data, waterux.data, hparam.gvnum.x*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hwateruy.data, wateruy.data, hparam.gvnum.y*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hwateruz.data, wateruz.data, hparam.gvnum.z*sizeof(float), cudaMemcpyDeviceToHost);

	CTimer time;
	time.startTimer();

	for (int idx = 0; idx<parNumNow; idx++)
	{
		float3 ipos = hpos[idx], ivel = hvel[idx];

		float3 gvel = getParticleVelFromGrid(ipos, hwaterux, hwateruy, hwateruz);

		float3 midpoint = ipos + gvel * hparam.dt * 0.5f;
		float3 gvelmidpoint = getParticleVelFromGrid(midpoint, hwaterux, hwateruy, hwateruz);
		ipos += gvelmidpoint * hparam.dt;

		hvel[idx] = ivel;
		hpos[idx] = ipos;
	}

	printTime(m_bCPURun, "advect_bubble_CPU", time);
}

void cspray::advect_bubble()
{

	if (m_bCPURun)
		advect_bubble_CPU();

	CTimer time;
	time.startTimer();
	advectparticle_RK2_bubble << <pblocknum, threadnum >> >(mParPos, mParVel, parNumNow,
		waterux, wateruy, wateruz, airux, airuy, airuz, hparam.dt, parflag, velmode);

	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
	printTime(m_bCPURun, "advect_bubble", time);
}

bool cspray::solver_cg_heat(charray A, farray x, farray b, int n)
{
	float a, zp, error2, eps = ((float)(1e-2))*n, alpha, beta;
	// z = applyA(x)
	computeAx_heat << <gsblocknum, threadnum >> >(z, A, x, n, HeatAlphaArray, fixedHeat, mscene);
	// r = b-Ax = b-1.0*z
	pcg_op_heat << <gsblocknum, threadnum >> >(A, r, b, z, -1.0, n);

	error2 = product(r, r, n);
	if (error2<eps)
		return true;

	//todo: applyPreconditioner
	//buildprecondition_pcg<<<gsblocknum,threadnum>>>( pre, A, z,r,n);
	cudaMemcpy(z.data, r.data, sizeof(float)*n, cudaMemcpyDeviceToDevice);	//z=r;

	cudaMemcpy(p.data, z.data, sizeof(float)*n, cudaMemcpyDeviceToDevice);	//p=z;

	a = product(z, r, n);	//a = z*r;

	int k = 0;
	for (; k<n; k++)
	{
		//for debug.
		if (k >= 3800)
		{
			if (!verifyfloat(a) || !verifyfloat(alpha) || !verifyfloat(error2))
				printf("there is some illegal float number in PCG solver!!!\n");
			printf("pcg iteration times: %d\n", k);
			mpause = true;
			break;
		}

		//z=A*p
		computeAx_heat << <gsblocknum, threadnum >> >(z, A, p, n, HeatAlphaArray, fixedHeat, mscene);

		zp = product(z, p, n);	//zp = z*p
		if (zp == 0)
			return true;

		alpha = a / zp;		//alpha = a/(z . p) = z*r/(z*p)

		// x = x + alpha*p
		pcg_op_heat << <gsblocknum, threadnum >> >(A, x, x, p, alpha, n);
		// r = r - alpha*z;
		pcg_op_heat << <gsblocknum, threadnum >> >(A, r, r, z, -alpha, n);

		//error2 = r * r
		error2 = product(r, r, n);
		if (error2<eps)
		{
			//	printf( "exit: k=%d,error2=%f\n", k,error2 );
			break;
		}

		//todo: applyPreconditioner
		cudaMemcpy(z.data, r.data, sizeof(float)*n, cudaMemcpyDeviceToDevice);	//z=r;
		//buildprecondition_pcg<<<gsblocknum,threadnum>>>( pre, A, z,r,n);

		//a2 = z*r;
		float a2 = product(z, r, n);

		beta = a2 / a;                     // beta = a2 / a
		// p = z + beta*p
		pcg_op_heat << <gsblocknum, threadnum >> >(A, p, z, p, beta, n);

		a = a2;
	}
	// 	if( mtime )
	// 		printf("CG interation: %d\n", k );

	return true;
}

void cspray::compTpChange_CPU()
{
	//	Tp, Tp_save, mmark 
	static farray htp, htpsave;
	static charray hmark;
	static bool first = true;

	if (first)
	{
		htp.setdim(NX, NY, NZ);
		htpsave.setdim(NX, NY, NZ);
		hmark.setdim(NX, NY, NZ);
		htp.data = new float[NX*NY*NZ];
		htpsave.data = new float[NX*NY*NZ];
		hmark.data = new char[NX*NY*NZ];
	}
	cudaMemcpy(htp.data, Tp.data, sizeof(float)*hparam.gnum, cudaMemcpyDeviceToHost);
	cudaMemcpy(htpsave.data, Tp_save.data, sizeof(float)*hparam.gnum, cudaMemcpyDeviceToHost);
	cudaMemcpy(hmark.data, mmark.data, sizeof(char)*hparam.gnum, cudaMemcpyDeviceToHost);


	CTimer time;
	time.startTimer();
	for (int i = 0; i<hparam.gnum; i++)
	{
		if (hmark[i] != TYPEBOUNDARY)
			htpsave[i] = htp[i] - htpsave[i];
		else
			htpsave[i] = 0;
	}
	printTime(true, "compTpChange_CPU", time);

	first = false;
}

void cspray::updateTemperature()
{
	//1. map heat from particle to grid
	mapHeatp2g_hash << <gsblocknum, threadnum >> >(mParPos, parTemperature, parNumNow, Tp, gridstart, gridend, defaulttemperature);
	cudaThreadSynchronize();

	//0. update the fixed heat with time.
	// 	updateFixedHeat<<<gsblocknum, threadnum>>> ( fixedHeat, mframe );
	// 	cudaThreadSynchronize();

	if (m_bAddHeatBottom)
	{
		addHeatAtBottom << <gsblocknum, threadnum >> >(Tp, mframe, heatIncreaseBottom);
		cudaThreadSynchronize();
	}
	//	setBoundaryHeat<<<gsblocknum, threadnum>>>( Tp );

	cudaMemcpy(Tp_save.data, Tp.data, sizeof(float)*hparam.gnum, cudaMemcpyDeviceToDevice);
	// 	printf("before cg heat:\n");
	// 	checkGridFarray( Tp );

	//2. set the right side of heat equation
	compb_heat << <gsblocknum, threadnum >> >(Tp_old, Tp, fixedHeat, mmark, HeatAlphaArray);
	cudaThreadSynchronize();

	// 	printf("compb_heat:\n");
	// 	checkGridFarray( Tp_old );

	//3. solve heat
	cudaMemset(Tp.data, 0, sizeof(float)*hparam.gnum);	//todo: 要不要这一步？？
	solver_cg_heat(mmark, Tp, Tp_old, hparam.gnum);
	cudaThreadSynchronize();

	//4. set boundary.
	if (m_bExtendHeatToBoundary)
		setBoundaryHeat << <gsblocknum, threadnum >> >(Tp);

	{
		if (m_bCPURun)
			compTpChange_CPU();

		CTimer time;
		time.startTimer();
		compTpChange << <gsblocknum, threadnum >> >(Tp, Tp_save, mmark);
		cudaThreadSynchronize();
		printTime(m_bCPURun, "compTpChange", time);
	}

	//5. map heat from grid to particle
	if (mscene == SCENE_MELTANDBOIL || mscene == SCENE_MELTANDBOIL_HIGHRES)
		mapHeatg2p_MeltAndBoil << <pblocknum, threadnum >> >(mParPos, parflag, parTemperature, parNumNow, Tp_save, Tp, defaultSolidT, alphaTempTrans);
	else
		mapHeatg2p << <pblocknum, threadnum >> >(mParPos, parflag, parTemperature, parNumNow, Tp_save, Tp, defaultSolidT, alphaTempTrans);

	cudaThreadSynchronize();

	updateLatentHeat();
	getLastCudaError("Kernel execution failed");
}

void cspray::initheat_grid()
{
	initheat_grid_k << <gsblocknum, threadnum >> >(Tp, mmark);
	cudaThreadSynchronize();
	markBoundaryCell << <gsblocknum, threadnum >> >(mmark);
	cudaThreadSynchronize();
}


//专门为了表现heat transfer而搭建的一个场景
void cspray::heatsim()
{
	if (!mpause)
	{
		CTimer time;
		time.startTimer();
		static CTimer timetotal;
		printTime(m_btimer, "TOTAL TIME!!", timetotal);
		printf("\n------------Frame %d:-------------\n", mframe);

		cudaMemcpy(Tp_save.data, Tp.data, sizeof(float)*hparam.gnum, cudaMemcpyDeviceToDevice);

		//2. set the right side of heat equation
		compb_heat << <gsblocknum, threadnum >> >(Tp_old, Tp, fixedHeat, mmark, HeatAlphaArray);
		cudaThreadSynchronize();

		//3. solve heat
		cudaMemset(Tp.data, 0, sizeof(float)*hparam.gnum);	//todo: 要不要这一步？？
		solver_cg_heat(mmark, Tp, Tp_old, hparam.gnum);
		cudaThreadSynchronize();

		//4. set boundary.
		if (m_bExtendHeatToBoundary)
			setBoundaryHeat << <gsblocknum, threadnum >> >(Tp);

		mframe++;
	}

}

void cspray::initTemperature()
{
	initHeatParticle << <pblocknum, threadnum >> >(parTemperature, parLHeat, defaultSolidT, defaultLiquidT, LiquidHeatTh, parflag, parNumNow);
	cudaThreadSynchronize();
}

void cspray::initSolubility()
{
	initsolubility_k << <pblocknum, threadnum >> >(parsolubility, pargascontain, parTemperature, parflag, parNumNow, 1.0f, Temperature0, initdissolvegasrate, initgasrate);
	cudaThreadSynchronize();
}

void cspray::computesurfacetension()
{
	compsurfacetension_k << <gsblocknum, threadnum >> >(surfacetension, mmark, phigrax, phigray, phigraz, surfacetensionsigma);
	cudaThreadSynchronize();

	enforcesurfacetension_p << <pblocknum, threadnum >> >(mParPos, mParVel, parflag, parNumNow, lsmerge, surfacetension, phigrax, phigray, phigraz, mmark, mscene);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}

void cspray::sweepLSAndMardGrid()
{
	//fluid
	//先sweep正值，再sweep负值
	markLS_bigpositive << <gsblocknum, threadnum >> >(lsfluid, mmark);
	cudaThreadSynchronize();
	for (int it = 0; it<5; it++)
	{
		sweepphibytype << <gsblocknum, threadnum >> >(lsfluid, mmark, TYPEFLUID);
		cudaThreadSynchronize();
	}
	setLSback_bigpositive << <gsblocknum, threadnum >> >(lsfluid);
	cudaThreadSynchronize();

	preparels << <gsblocknum, threadnum >> >(lsfluid, mmark);
	cudaThreadSynchronize();
	for (int it = 0; it<5; it++)
	{
		sweepphibytype << <gsblocknum, threadnum >> >(lsfluid, mmark, TYPEFLUID);
		cudaThreadSynchronize();
	}
	setLSback << <gsblocknum, threadnum >> >(lsfluid);
	cudaThreadSynchronize();

	//air
	//先sweep正值，再sweep负值
	markLS_bigpositive << <gsblocknum, threadnum >> >(lsair, mmark);
	cudaThreadSynchronize();
	for (int it = 0; it<5; it++)
	{
		sweepphibytype << <gsblocknum, threadnum >> >(lsair, mmark, TYPEFLUID);
		cudaThreadSynchronize();
	}
	setLSback_bigpositive << <gsblocknum, threadnum >> >(lsair);
	cudaThreadSynchronize();

	preparels << <gsblocknum, threadnum >> >(lsair, mmark);
	cudaThreadSynchronize();
	for (int it = 0; it<5; it++)
	{
		sweepphibytype << <gsblocknum, threadnum >> >(lsair, mmark, TYPEFLUID);
		cudaThreadSynchronize();
	}
	setLSback << <gsblocknum, threadnum >> >(lsair);
	cudaThreadSynchronize();

	//merge the level set.
	mergeLSAndMarkGrid << <gsblocknum, threadnum >> >(lsmerge, mmark, lsfluid, lsair);
	cudaThreadSynchronize();

	//计算距离场的梯度
	computePhigra << <gsblocknum, threadnum >> >(phigrax, phigray, phigraz, lsmerge);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
}

void cspray::correctpos_bubble()
{
	static bool first = true;
	static float *dphi;
	//	char *hflag;
	if (first)
	{
		cudaMalloc((void**)&dphi, sizeof(float)*parNumMax);
		// 		hphi=new float[parNumMax];
		// 		hflag = new char[parNumMax];
		first = false;
	}
	cudaMemset(dphi, 0, parNumNow*sizeof(float));


	if (mscene == SCENE_MELTANDBOIL || mscene == SCENE_MELTANDBOIL_HIGHRES || mscene == SCENE_INTERACTION)
	{
		computePhigra << <gsblocknum, threadnum >> >(phigrax_air, phigray_air, phigraz_air, lsair);
		correctbubblepos_air << <pblocknum, threadnum >> >(lsmerge, phigrax, phigray, phigraz, lsair, phigrax_air, phigray_air, phigraz_air,
			mParPos, parflag, parNumNow, dphi);
	}
	else
		correctbubblepos << <pblocknum, threadnum >> >(lsmerge, phigrax, phigray, phigraz, mParPos, parflag, parNumNow, dphi);


	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	//	first=false;
	// 	cudaMemcpy( hphi, dphi, parNumNow*sizeof(float), cudaMemcpyDeviceToHost );
	// 	cudaMemcpy( hflag, parflag, parNumNow*sizeof(char), cudaMemcpyDeviceToHost );
	// 	float aver=0, maxphi=-100000, minphi=1000000;
	// 	for( int i=0; i<parNumNow; i++ )
	// 	{
	// 		float t=hphi[i]/hparam.cellsize.x;
	// 		if( hflag[i]==TYPEAIR )
	// 			printf( "phi=%f\n", t ), maxphi=max(maxphi,t), minphi=min(minphi,t);
	// 	}
	// 	printf( "%f, %f\n", maxphi, minphi );
	// 
}

void cspray::initscene_fluidsphere()
{
	if (parNumNow <= 0)
		return;

	float3* hparpos = new float3[parNumNow];
	float3* hparvel = new float3[parNumNow];
	float* hparmass = new float[parNumNow];
	char* hparflag = new char[parNumNow];

	float3 bubblepos1 = make_float3(NX*0.5f, NY*0.5f, NZ*0.4f) * hparam.cellsize.x;		//bottom
	float3 bubblepos2 = make_float3(NX*0.5f, NY*0.5f, NZ*0.6f) * hparam.cellsize.x;		//bottom
	float3 bubblepos3 = make_float3(NX*0.5f, NY*0.5f, NZ*0.75f) * hparam.cellsize.x;		//bottom
	float3 bubblepos4 = make_float3(NX*0.3f, NY*0.5f, NZ*0.15f) * hparam.cellsize.x;		//bottom
	float3 bubblepos5 = make_float3(NX*0.7f, NY*0.5f, NZ*0.15f) * hparam.cellsize.x;		//bottom
	float bubbleradius = NX*0.5f*hparam.cellsize.x * 0.3f;		//small bubble
	float3 temppos;

	int i = 0;
	for (float z = hparam.cellsize.x + hparam.samplespace; z< (NZ - 2)*hparam.cellsize.x && i<initfluidparticle; z += hparam.samplespace)
	for (float y = hparam.cellsize.x + hparam.samplespace; y<hparam.cellsize.x*(NY - 1) - 0.5f*hparam.samplespace && i<initfluidparticle; y += hparam.samplespace)
	for (float x = hparam.cellsize.x + hparam.samplespace; x<hparam.cellsize.x*(NX - 1) - 0.5f*hparam.samplespace && i<initfluidparticle; x += hparam.samplespace)
	{
		temppos = make_float3(0.4f*(x - bubblepos1.x), y - bubblepos1.y, z - bubblepos1.z);
		if (length(temppos)<bubbleradius)
		{
			hparflag[i] = TYPEFLUID;
			hparpos[i] = make_float3(x, y, z);
			hparvel[i] = make_float3(0.0f);
			hparmass[i] = hparam.m0;
			i++;
		}
		temppos = make_float3(0.6f*(x - bubblepos2.x), y - bubblepos2.y, z - bubblepos2.z);
		if (length(temppos)<bubbleradius)
		{
			hparflag[i] = TYPEFLUID;
			hparpos[i] = make_float3(x, y, z);
			hparvel[i] = make_float3(0.0f);
			hparmass[i] = hparam.m0;
			i++;
		}
		temppos = make_float3(1.0f*(x - bubblepos3.x), y - bubblepos3.y, z - bubblepos3.z);
		if (length(temppos)<bubbleradius)
		{
			hparflag[i] = TYPEFLUID;
			hparpos[i] = make_float3(x, y, z);
			hparvel[i] = make_float3(0.0f);
			hparmass[i] = hparam.m0;
			i++;
		}
		temppos = make_float3((x - bubblepos4.x), y - bubblepos4.y, z - bubblepos4.z)*0.75f;
		if (length(temppos)<bubbleradius)
		{
			hparflag[i] = TYPEFLUID;
			hparpos[i] = make_float3(x, y, z);
			hparvel[i] = make_float3(0.0f);
			hparmass[i] = hparam.m0;
			i++;
		}
		temppos = make_float3((x - bubblepos5.x), y - bubblepos5.y, z - bubblepos5.z)*0.75f;
		if (length(temppos)<bubbleradius)
		{
			hparflag[i] = TYPEFLUID;
			hparpos[i] = make_float3(x, y, z);
			hparvel[i] = make_float3(0.0f);
			hparmass[i] = hparam.m0;
			i++;
		}
	}

	parNumNow = i;
	//debug:
	//hparpos[parNumNow-1] = make_float3(1-1.2f/64,1.2f/64, 1-1.2f/64 );

	cudaMemcpy(mParPos, hparpos, sizeof(float3)*parNumNow, cudaMemcpyHostToDevice);
	cudaMemcpy(mParVel, hparvel, sizeof(float3)*parNumNow, cudaMemcpyHostToDevice);
	cudaMemcpy(parmass, hparmass, sizeof(float)*parNumNow, cudaMemcpyHostToDevice);
	cudaMemcpy(parflag, hparflag, sizeof(char)*parNumNow, cudaMemcpyHostToDevice);

	delete[] hparpos;
	delete[] hparvel;
	delete[] hparmass;
	delete[] hparflag;
}

void cspray::deleteAirFluidParticle()
{
	//1. sweep vacuum, mark the air cell as vacuum if it adjoins a vacuum cell.
	if (mscene == SCENE_INTERACTION)
	{

		for (int t = 0; t<1 && mframe % 2 == 0; t++)
			sweepVacuum << <gsblocknum, threadnum >> >(mmark);
	}
	else
	{
		for (int t = 0; t<20; t++)
			sweepVacuum << <gsblocknum, threadnum >> >(mmark);
	}

	//2. mark the deleted particle
	uint *cnt;
	cudaMalloc((void**)&cnt, sizeof(uint));
	cudaMemset(cnt, 0, sizeof(uint));
	markDeleteAirParticle << <pblocknum, threadnum >> >(mParPos, parflag, parmass, preservemark, parNumNow, mmark, lsmerge, lsair, cnt);
	uint *hcnt = new uint[1];
	cudaMemcpy(hcnt, cnt, sizeof(uint), cudaMemcpyDeviceToHost);
	static uint totalCnt = 0;
	totalCnt += hcnt[0];
	printf("totalCnt for deleting fluid particle=%u\n", totalCnt);

	//3. delete particle.
	//3. scan，得到需要保留的粒子的最终序号
	int activeParticleNum = 0;
	ThrustScanWrapper(preservemarkscan, preservemark, parNumNow);
	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *)&lastElement,
			(void *)(preservemark + parNumNow - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
			(void *)(preservemarkscan + parNumNow - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		activeParticleNum = lastElement + lastScanElement;
	}

	if (activeParticleNum == parNumNow)	//如果没有需要删除的粒子
		return;

	//4. copy，把粒子的质量、速度、标记、位置拷贝过来。
	swapParticlePointers();
	deleteparticles << <pblocknum, threadnum >> >(preservemark, preservemarkscan, parNumNow,
		mParPos, tmpParPos, mParVel, tmpParVelFLIP, parmass, tmpparmass, parflag, tmpparflag, parTemperature, tmpparTemperature,
		parLHeat, tmpparHeat, parsolubility, tempsolubility, pargascontain, tempgascontain);

	//5. 修改粒子的个数，并改pblocknum
	parNumNow = activeParticleNum;
	pblocknum = max(1, (int)ceil(((float)parNumNow) / threadnum));
	printf("After deleting: particle number:%d\n", parNumNow);
}

void cspray::outputPovRaywater(int frame, float3* dpos, float3 *dnormal, int pnum, uint *dindices, int indicesnum, const char* objectname)
{
	//filename
	static char filename[100];
	sprintf(filename, "%swaterdata\\%s%05d.pov", outputdir, objectname, frame);
	FILE *fp = fopen(filename, "w");
	if (fp == NULL)
	{
		printf("cannot open pov file for output!!\n");
		mpause = true;
		return;
	}

	//如果没有MC三角形，则只引用头文件且输出
	if (pnum == 0)
	{
		fclose(fp);
		return;
	}

	fprintf(fp, "#declare watermesh=mesh2{\n");
	fprintf(fp, "vertex_vectors{\n");
	fprintf(fp, "%d,\n", pnum);

	// vertex positions
	static float3* hpos = new float3[maxVerts];
	cudaMemcpy(hpos, dpos, pnum*sizeof(float3), cudaMemcpyDeviceToHost);
	for (int i = 0; i<pnum; i++)
	{
		if (!verifyfloat3(hpos[i]))
			hpos[i] = make_float3(0.0f);
		fprintf(fp, "< %f, %f, %f >,\n", hpos[i].x, hpos[i].y, hpos[i].z);
	}
	fprintf(fp, "}\n ");

	fprintf(fp, "normal_vectors{\n");
	fprintf(fp, "%d,\n", pnum);
	//vertex normals
	cudaMemcpy(hpos, dnormal, pnum*sizeof(float3), cudaMemcpyDeviceToHost);
	for (int i = 0; i<pnum; i++)
	{
		if (!verifyfloat3(hpos[i]))
			hpos[i] = make_float3(0.0f);
		fprintf(fp, "<%f, %f, %f>,\n", hpos[i].x, hpos[i].y, hpos[i].z);
	}
	fprintf(fp, "}\n ");

	fprintf(fp, "face_indices{\n");
	fprintf(fp, "%d,\n", indicesnum / 3);
	//face indices.
	static uint *hindices = new uint[MCedgeNum];
	cudaMemcpy(hindices, dindices, indicesnum*sizeof(uint), cudaMemcpyDeviceToHost);
	for (int i = 0; i<indicesnum; i += 3)
		fprintf(fp, "<%u, %u, %u>,\n", hindices[i], hindices[i + 1], hindices[i + 2]);	//povray indices from 0 !!!
	fprintf(fp, "}\n ");

	fprintf(fp, "inside_vector <0,0,1> }\n ");
	fprintf(fp, "object{ watermesh material{%s_material} }\n ", objectname);

	fclose(fp);
}
void cspray::outputColoredParticle(int frame, float3* dpos, float *dtemperature, int pnum)
{
	static float3 *hpos = new float3[parNumMax];
	static float *htemperature = new float[parNumMax];

	cudaMemcpy(hpos, dpos, pnum*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(htemperature, dtemperature, pnum*sizeof(float), cudaMemcpyDeviceToHost);

	//filename
	static char filename[100];
	sprintf(filename, "%swaterdata\\allparticles%05d.pov", outputdir, frame);
	FILE *fp = fopen(filename, "w");
	if (fp == NULL)
	{
		printf("cannot open pov file for output!!\n");
		return;
	}

	float iradius = 0.004f;
	float3 color;
	for (int i = 0; i<pnum; i++)
	{
		color = mapColorBlue2Red_h((htemperature[i] - temperatureMin_render) / (temperatureMax_render - temperatureMin_render)*6.0f);
		fprintf(fp, "sphere{ <%f,%f,%f> %f texture{ finish{dropletFinish} pigment{ rgb<%f,%f,%f>}} }\n", hpos[i].x, hpos[i].y, hpos[i].z, iradius, color.x, color.y, color.z);
	}

	fclose(fp);
}

void cspray::outputSoloBubblePovRay(int frame, float3 *dpos, float *dmass, char *dflag, int pnum)
{
	static float3 *hpos = new float3[parNumMax];
	//	static float *hmass = new float[parNumMax];
	static char *hflag = new char[parNumMax];

	cudaMemcpy(hpos, dpos, pnum*sizeof(float3), cudaMemcpyDeviceToHost);
	//	cudaMemcpy( hmass, dmass, pnum*sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy(hflag, dflag, pnum*sizeof(char), cudaMemcpyDeviceToHost);

	//filename
	static char filename[100];
	sprintf(filename, "%swaterdata\\solobubble%05d.pov", outputdir, frame);
	FILE *fp = fopen(filename, "w");
	if (fp == NULL)
	{
		printf("cannot open pov file for output!!\n");
		return;
	}

	float iradius = 0.005f;
	for (int i = 0; i<pnum; i++)
	{
		if (hflag[i] == TYPEAIRSOLO)
			fprintf(fp, "sphere{ <%f,%f,%f> %f texture{bblTexture} }\n", hpos[i].x, hpos[i].y, hpos[i].z, iradius);
	}

	fclose(fp);
}

void cspray::outputAirParticlePovRay(int frame, float3 *dpos, float *dmass, char *dflag, int pnum)
{
	static float3 *hpos = new float3[parNumMax];
	//	static float *hmass = new float[parNumMax];
	static char *hflag = new char[parNumMax];

	cudaMemcpy(hpos, dpos, pnum*sizeof(float3), cudaMemcpyDeviceToHost);
	//	cudaMemcpy( hmass, dmass, pnum*sizeof(float), cudaMemcpyDeviceToHost );
	cudaMemcpy(hflag, dflag, pnum*sizeof(char), cudaMemcpyDeviceToHost);

	//filename
	static char filename[100];
	sprintf(filename, "%swaterdata\\solobubble%05d.pov", outputdir, frame);
	FILE *fp = fopen(filename, "w");
	if (fp == NULL)
	{
		printf("cannot open pov file for output!!\n");
		return;
	}

	float iradius = 0.005f;
	for (int i = 0; i<pnum; i++)
	{
		if (hflag[i] == TYPEAIR)
			fprintf(fp, "sphere{ <%f,%f,%f> %f texture{bblTexture} }\n", hpos[i].x, hpos[i].y, hpos[i].z, iradius);
	}

	fclose(fp);
}

void cspray::outputEmptyBubblePovRay(int frame)
{
	//filename
	static char filename[100];
	sprintf(filename, "%swaterdata\\emptybubble%05d.pov", outputdir, frame);
	FILE *fp = fopen(filename, "w");
	if (fp == NULL)
	{
		printf("cannot open pov file for output!!\n");
		return;
	}

	static float3 *hEmptyPos = new float3[pEmptyNum];
	static float3 *hEmptyDir = new float3[pEmptyNum];
	static float *hEmptyRadius = new float[pEmptyNum];
	cudaMemcpy(hEmptyPos, pEmptyPos, pEmptyNum*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(hEmptyDir, pEmptyDir, pEmptyNum*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaMemcpy(hEmptyRadius, pEmptyRadius, pEmptyNum*sizeof(float), cudaMemcpyDeviceToHost);
	printf("emptybubble pos=%f,%f,%f, radius=%f\n", hEmptyPos[0].x, hEmptyPos[0].y, hEmptyPos[0].z, hEmptyRadius[0]);

	//	float iradius=0.005f;
	for (int i = 0; i<pEmptyNum; i++)
	{
		//if( hflag[i]==TYPEAIRSOLO)
		fprintf(fp, "sphere{ <%f,%f,%f> %f texture{bblTexture} }\n", hEmptyPos[i].x, hEmptyPos[i].y, hEmptyPos[i].z, hEmptyRadius[0]);
	}

	fclose(fp);
}


void cspray::initscene_multibubble()
{
	if (parNumNow <= 0)
		return;

	float3* hparpos = new float3[parNumNow];
	float3* hparvel = new float3[parNumNow];
	float* hparmass = new float[parNumNow];
	char* hparflag = new char[parNumNow];

	float bubbleradius = hparam.cellsize.x*3.0f;		//small bubble
	const int bubblecnt = 3;
	float3 bubblepos[bubblecnt];
	int k = 0;
	for (int i = 1; i <= 3; ++i)
		bubblepos[k++] = make_float3(i / 4.0f*NX, 0.5f*NY, 5.0f)*hparam.cellsize.x;
	float3 temppos;

	int i = 0;
	for (float z = hparam.cellsize.x + hparam.samplespace; z< (NZ - 2)*hparam.cellsize.x && i<initfluidparticle; z += hparam.samplespace)
	{
		for (float y = hparam.cellsize.x + hparam.samplespace; y<hparam.cellsize.x*(NY - 1) - 0.5f*hparam.samplespace && i<initfluidparticle; y += hparam.samplespace)
		for (float x = hparam.cellsize.x + hparam.samplespace; x<hparam.cellsize.x*(NX - 1) - 0.5f*hparam.samplespace && i<initfluidparticle; x += hparam.samplespace)
		{
			bool flag = false;
			for (int cnt = 0; cnt<bubblecnt; ++cnt)
			{
				temppos = make_float3(x, y, z);
				if (length(temppos - bubblepos[cnt])<bubbleradius)
					flag = true;
			}
			if (flag)
				hparflag[i] = TYPEAIRSOLO;
			else
				hparflag[i] = TYPEFLUID;
			hparpos[i] = make_float3(x, y, z);
			hparvel[i] = make_float3(0.0f);
			hparmass[i] = hparam.m0;
			i++;
		}
	}

	parNumNow = i;
	//debug:
	//hparpos[parNumNow-1] = make_float3(1-1.2f/64,1.2f/64, 1-1.2f/64 );

	cudaMemcpy(mParPos, hparpos, sizeof(float3)*parNumNow, cudaMemcpyHostToDevice);
	cudaMemcpy(mParVel, hparvel, sizeof(float3)*parNumNow, cudaMemcpyHostToDevice);
	cudaMemcpy(parmass, hparmass, sizeof(float)*parNumNow, cudaMemcpyHostToDevice);
	cudaMemcpy(parflag, hparflag, sizeof(char)*parNumNow, cudaMemcpyHostToDevice);

	delete[] hparpos;
	delete[] hparvel;
	delete[] hparmass;
	delete[] hparflag;
}

void cspray::markSoloAirParticle()
{
	verifySoloAirParticle << <pblocknum, threadnum >> >(mParPos, mParVel, parflag, parNumNow, lsmerge, airux, airuy, airuz, gridstart, gridend, mscene);
	cudaThreadSynchronize();
}

void cspray::updateSoloAirParticle()
{
	calcDensPress_Air << <pblocknum, threadnum >> >(mParPos, pardens, parpress, parflag, parNumNow, gridstart, gridend);
	cudaThreadSynchronize();

	//debug
	if (0)
	{
		static float *hdens = new float[parNumMax];
		static char *hflag = new char[parNumMax];
		cudaMemcpy(hdens, pardens, parNumNow*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(hflag, parflag, parNumNow*sizeof(char), cudaMemcpyDeviceToHost);

		float densaver = 0, densmax = -1, densmin = 10000000;
		int cnt = 0;
		for (int i = 0; i<parNumNow; i++)
		{
			if (hflag[i] == TYPEAIRSOLO) {
				densaver += 1 / hdens[i];
				densmax = max(densmax, 1 / hdens[i]);
				densmin = min(densmin, 1 / hdens[i]);
				cnt++;
				printf("%f, ", 1 / hdens[i]);
			}
		}
		densaver /= cnt;
		printf("densaver, max, min=%f,%f,%f\n", densaver, densmax, densmin);
	}

	enforceForceSoloAirP << <pblocknum, threadnum >> >(mParPos, mParVel, pardens, parpress, parflag, parNumNow, gridstart, gridend, viscosiySPH, maxVelForBubble);
	cudaThreadSynchronize();
}

void cspray::GenerateGasParticle()
{
	//1. 根据温度改变每个粒子的溶解度
	updatesolubility << <pblocknum, threadnum >> >(parsolubility, parTemperature, parflag, parNumNow, 1.0f, Temperature0, initdissolvegasrate);
	cudaThreadSynchronize();

	static bool first = true;

	static int *daddparnums;
	int haddparnums = 0;
	//	static bool first=true;
	if (first)
	{
		cudaMalloc((void**)&daddparnums, sizeof(int));
		first = false;
	}
	cudaMemcpy(daddparnums, &haddparnums, sizeof(int), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();

	//3. 根据溶解度和latent heat生成气体粒子
	GenerateGasParticle_k << <gsblocknum, threadnum >> >(parsolubility, pargascontain, mParPos, mParVel, parmass, parflag, parTemperature, parLHeat, parNumNow, gridstart, gridend,
		daddparnums, randfloat, randfloatcnt, mframe, Tp, LiquidHeatTh, dseedcell, seednum, vaporGenRate);

	//update the number of particles.
	cudaMemcpy(&haddparnums, daddparnums, sizeof(int), cudaMemcpyDeviceToHost);
	parNumNow += haddparnums;
	pblocknum = max(1, (int)ceil(((float)parNumNow) / threadnum));
	//printf("addparnums=%d\n", haddparnums );

	//printf( "After adjustAirParBySolubility, particle number=%d\n", parNumNow );
}

void cspray::solidmotion_fixed()
{
	setVelZeroSolid_k << <pblocknum, threadnum >> >(mParVel, parflag, parNumNow);
	cudaThreadSynchronize();
}

void cspray::solidmotion()			///////////////////////////////
{
	//若没有固体粒子则直接返回
	if (nRealSolpoint <= 0)
		return;

	//	printf("nrealSolPoint=%d\n", nRealSolpoint );

	//0. 处理刚体碰到墙的情况
	solidCollisionWithBound << <pblocknum, threadnum >> > (mParPos, mParVel, parflag, parNumNow, SolidbounceParam, nRealSolpoint);
	cudaThreadSynchronize();

	//1. prepare
	cudaMemcpy(solidParPos, mParPos, sizeof(float3)*parNumNow, cudaMemcpyDeviceToDevice);
	cudaMemcpy(solidParVelFLIP, mParVel, sizeof(float3)*parNumNow, cudaMemcpyDeviceToDevice);

	set_nonsolid_2_zero << <pblocknum, threadnum >> >(parflag, parNumNow, solidParPos, solidParVelFLIP);	//	非固体粒子速度和位置置0.就可以直接累加了
	cudaThreadSynchronize();

	float3 T, tmpI;	//T: 线速度；tmpI：每个粒子到重心的距离
	float3 R, rg;		//R:角速度；rg: 重心

	R = T = rg = make_float3(0, 0, 0);
	float fSolNum = (float)nRealSolpoint;

	//2. 求刚体的线速度
	T = accumulate_GPU_f3(solidParVelFLIP);
	//	printf("T=%f,%f,%f\n", T.x, T.y, T.z );
	T = T / fSolNum;

	//interaction场景里y不变化，方便观察固体与气泡的交互
	if (mscene == SCENE_INTERACTION)
		T.y = 0.0f;

	//3. 求刚体的重心
	rg0 = accumulate_GPU_f3(solidParPos) / fSolNum;
	rg = rg0 + hparam.dt*T;					//更新重心位置

	//4. 求刚体的角速度
	compute_cI_k << <pblocknum, threadnum >> >(parNumNow, parflag, mParPos, mParVel, c, I, rg0);
	tmpI = accumulate_GPU_f3(I);	//各粒子到重心的距离之和，相当于角速度的权重
	R = accumulate_GPU_f3(c);		//角速度（带权重）
	if (tmpI.x <= 0)	tmpI.x = 1;
	//这里的步骤有些意思：tmpI.x的概念是全部粒子到重心距离的平均值，而R是各粒子“角速度(没有除|r|^2)”的平均值。
	//这样计算相比针对每个粒子各自计算角速度要更“正确”，可以测试一下只加重力向下除的场景。
	tmpI.x /= fSolNum;
	R = R / tmpI.x / tmpI.x;		//归一化
	R /= fSolNum;
	//	printf( "tempi.x=%f\n", tmpI.x );

	//printf( "rg=%f,%f,%f, R=%f,%f,%f, T=%f,%f,%f\n", rg.x, rg.y, rg.z, R.x, R.y, R.z, T.x, T.y, T.z );

	//5. 更新各粒子的速度
	computeVelSolid_k << <pblocknum, threadnum >> >(mParPos, parflag, mParVel, parNumNow, rg0, R, T);
	cudaThreadSynchronize();

	//6. 更新各粒子的位置
	{
		//debug
		//R=make_float3( 0,0,1 );
		float3 axis = R;		//角速度决定的旋转轴
		float theta = -length(R)*hparam.dt;		//角速度决定的旋转角度大小
		if (abs(theta)>1e-6)		//注意：轴长为0时是不能normalize的，会有除0的错误
			axis = normalize(axis);
		else
			axis = make_float3(1, 0, 0), theta = 0;

		matrix3x3 rm;		//旋转矩阵，由旋转轴axis与旋转角度theta决定，公式参考：http://zh.wikipedia.org/wiki/%E6%97%8B%E8%BD%AC%E7%9F%A9%E9%98%B5
		float x = axis.x, y = axis.y, z = axis.z;
		//printf("axis=%f,%f,%f, theta=%f\n", x,y,z,theta);
		float c = cos(theta), s = sin(theta);
		rm.x00 = c + (1 - c)*x*x, rm.x01 = (1 - c)*x*y - s*z, rm.x02 = (1 - c)*x*z + s*y;
		rm.x10 = (1 - c)*y*x + s*z, rm.x11 = c + (1 - c)*y*y, rm.x12 = (1 - c)*y*z - s*x;
		rm.x20 = (1 - c)*z*x - s*y, rm.x21 = (1 - c)*z*y + s*x, rm.x22 = c + (1 - c)*z*z;
		//normalize? not needed.

		computePosSolid_k << <pblocknum, threadnum >> >(mParPos, parflag, parNumNow, rg, rg0, rm);
		cudaThreadSynchronize();

		int blocknum = (int)ceil(((float)solidvertexnum) / threadnum);
		if ((mscene == SCENE_INTERACTION || mscene == SCENE_INTERACTION_HIGHRES || mscene == SCENE_MELTANDBOIL || mscene == SCENE_MELTANDBOIL_HIGHRES) && mframe>0 && !bRunMCSolid)
			computeSolidVertex_k << <blocknum, threadnum >> >(solidvertex, solidvertexnum, rg, rg0, rm);
	}
}

//parNumNow个粒子的float3类型的数据的归约求和
float3 cspray::accumulate_GPU_f3(float3 *data)
{
	int maxblockNum = max(1, (int)ceil(((float)parNumMax) / threadnum));
	static float3 *ssum = NULL;
	if (!ssum)
		cudaMalloc((void**)&ssum, sizeof(float3)*maxblockNum/*GRIDCOUNT(60160,256)*/);
	static float3*hsum = new float3[maxblockNum];
	int sharememsize = threadnum*sizeof(float3);

	accumulate_GPU_k << <pblocknum, threadnum, sharememsize >> >(parNumNow, ssum, data);
	cudaThreadSynchronize();

	cudaMemcpy(hsum, ssum, sizeof(float3)*pblocknum, cudaMemcpyDeviceToHost);

	float3 res = make_float3(0);
	for (int i = 0; i<pblocknum; i++)
	{
		res += hsum[i];
		//	printf( "debug: hsum = %f,%f,%f,%i\n", hsum[i].x,hsum[i].y,hsum[i].z,i);
	}
	return res;
}

//parNumNow个粒子的float3类型的数据的归约求和
float3 cspray::accumulate_CPU_f3_test(float3 *data)
{
	float3 res = make_float3(0);
	static float3 *hdata = new float3[parNumMax];
	cudaMemcpy(hdata, data, sizeof(float3)*parNumNow, cudaMemcpyDeviceToHost);

	for (int i = 0; i<parNumNow; i++)
	{
		res += hdata[i];
		//	printf( "debug: hsum = %f,%f,%f,%i\n", hsum[i].x,hsum[i].y,hsum[i].z,i);
	}
	return res;
}


void cspray::MeltSolid()
{
	static int *dnumchange = NULL;
	if (!dnumchange)
		cudaMalloc((void**)&dnumchange, sizeof(int));
	int hnumchange = 0;
	cudaMemset(dnumchange, 0, sizeof(int));
	cudaThreadSynchronize();

	MeltingSolidByHeat << <pblocknum, threadnum >> > (parTemperature, parLHeat, parflag, parNumNow, LiquidHeatTh, meltingpoint, dnumchange);
	cudaThreadSynchronize();

	cudaMemcpy((void*)&hnumchange, dnumchange, sizeof(int), cudaMemcpyDeviceToHost);
	nRealSolpoint -= hnumchange;
	//printf( "the new solid point number = %d\n", nRealSolpoint );
}

void cspray::MeltSolid_CPU()
{
	cudaMemcpy(hparflag, parflag, sizeof(char)*parNumNow, cudaMemcpyDeviceToHost);
	cudaMemcpy(hparLHeat, parLHeat, sizeof(float)*parNumNow, cudaMemcpyDeviceToHost);

	CTimer time;
	time.startTimer();
	int hnumchange = 0;

	for (int i = 0; i<parNumNow; i++)
	{
		if (hparflag[i] == TYPESOLID)
		{
			if (hparLHeat[i]>LiquidHeatTh)
			{
				hparflag[i] = TYPEFLUID;
				hparLHeat[i] = LiquidHeatTh;
				hnumchange = 0;
			}
		}
	}

	nRealSolpoint -= hnumchange;
	//printf( "the new solid point number = %d\n", nRealSolpoint );
	printTime(true, "MeltSolid_CPU", time);
}

void cspray::Freezing()
{
	static int *dnumchange = NULL;
	if (!dnumchange)
		cudaMalloc((void**)&dnumchange, sizeof(int));
	int hnumchange = 0;
	cudaMemset(dnumchange, 0, sizeof(int));
	cudaThreadSynchronize();

	FreezingSolidByHeat << <pblocknum, threadnum >> > (mParPos, parLHeat, parflag, parNumNow, dnumchange, gridstart, gridend);
	cudaThreadSynchronize();

	cudaMemcpy((void*)&hnumchange, dnumchange, sizeof(int), cudaMemcpyDeviceToHost);
	nRealSolpoint += hnumchange;
	//printf( "Freezing: new solid point number=%d\n", nRealSolpoint );
}

void cspray::initEmptyBubbles()
{
	pEmptyNum = 1;
	float3 *hEmptyPos = new float3[pEmptyNum];
	float3 *hEmptyDir = new float3[pEmptyNum];
	float *hEmptyRadius = new float[pEmptyNum];

	for (int i = 0; i<pEmptyNum; i++)
	{
		hEmptyPos[i] = make_float3(12.f*hparam.cellsize.x, 12.f*hparam.cellsize.x, 1.6f*hparam.cellsize.x);
		hEmptyDir[i] = make_float3(0, 0, 1);
		hEmptyRadius[i] = 0;
	}
	printf("emptybubble pos=%f,%f,%f, radius=%f\n", hEmptyPos[0].x, hEmptyPos[0].y, hEmptyPos[0].z, hEmptyRadius[0]);

	//for empty气泡的生成
	cudaMalloc((void**)&pEmptyPos, sizeof(float3)*pEmptyNum);
	cudaMalloc((void**)&pEmptyDir, sizeof(float3)*pEmptyNum);
	cudaMalloc((void**)&pEmptyRadius, sizeof(float)*pEmptyNum);

	cudaMemcpy(pEmptyPos, hEmptyPos, pEmptyNum*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(pEmptyDir, hEmptyDir, pEmptyNum*sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(pEmptyRadius, hEmptyRadius, pEmptyNum*sizeof(float), cudaMemcpyHostToDevice);

	delete[] hEmptyPos;
	delete[] hEmptyDir;
	delete[] hEmptyRadius;
}


void cspray::initSeedCell()
{
	//随机x,y的值，然后计算格子的值
	int *hseedcells = new int[seednum];
	int z = 2;
	for (int i = 0; i<seednum; i++)
	{
		bool has = true;	//记录现在生成的cell是否已经被用过
		while (has)
		{
			has = false;
			int x = rand() % (NX - 4) + 2;
			int y = rand() % (NY - 4) + 2;
			int gidx = getidx(x, y, z);
			for (int j = 0; j<i; j++)
			if (hseedcells[j] == gidx) has = true;
			if (!has)
			{
				hseedcells[i] = gidx;
				//	printf("x,y=%d,%d\n", x, y);
			}
		}
	}
	cudaMemcpy(dseedcell, hseedcells, sizeof(int)*seednum, cudaMemcpyHostToDevice);

	delete[] hseedcells;
}

void cspray::updateSeedCell()
{
	static int idx = 0;

	int cnt = 0, bound = 3;
	while (cnt++ <= 3)
	{
		int z = 2;
		int x = rand() % (NX - 2 * bound) + bound;
		int y = rand() % (NY - 2 * bound) + bound;
		int gidx = getidx(x, y, z);

		cudaMemcpy(dseedcell + idx, &gidx, sizeof(int), cudaMemcpyHostToDevice);
		idx++;
		idx %= seednum;
	}
}

void cspray::enforceDragForce()
{
	calDragForce << < pblocknum, threadnum >> >(mParPos, mParVel, parflag, parNumNow, waterux, wateruy, wateruz, dragParamSolo, dragParamGrid, mscene);
	cudaThreadSynchronize();
}

void cspray::CollisionSolid()
{
	//1. prepare phi and velocity of solid (only solid, this is for collision)
	initSolidPhi << <gsblocknum, threadnum >> >(phisolid, gridstart, gridend, parflag);
	cudaThreadSynchronize();

	for (int it = 0; it<3; it++)
	{
		sweepphi << <gsblocknum, threadnum >> >(phisolid);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed");
	}
	//1.2, velocity
	mapvelp2g_k_fluidSolid << <gvblocknum, threadnum >> >(mParPos, mParVel, parmass, parflag, parNumNow, solidux, soliduy, soliduz, gridstart, gridend);
	cudaThreadSynchronize();

	//todo: sweep u
	//2. modify the velocity of air/airsolo/fluid particle, and update their positions.
	if (mscene == SCENE_FREEZING || mscene == SCENE_MELTINGPOUR)
		CollisionWithSolid_Freezing << <pblocknum, threadnum >> > (mParPos, mParVel, parflag, parNumNow, phisolid, gridstart, gridend);
	else
		CollisionWithSolid_k << <pblocknum, threadnum >> > (mParPos, mParVel, parflag, parNumNow, phisolid, solidux, soliduy, soliduz, mscene, bounceVelParam, bouncePosParam);

	cudaThreadSynchronize();
}

void cspray::updateLatentHeat()
{
	updateLatentHeat_k << <pblocknum, threadnum >> >(parTemperature, parLHeat, parflag, parNumNow, meltingpoint, boilingpoint, LiquidHeatTh);
	cudaThreadSynchronize();
}

void cspray::genAirFromSolid()
{
	// 	static int *daddparnums;
	// 	int haddparnums=0;
	// 	static bool first=true;
	// 	if(first)
	// 	{
	// 		cudaMalloc( (void**)&daddparnums, sizeof(int));
	// 		first=false;
	// 	}
	// 	cudaMemcpy( daddparnums, &haddparnums, sizeof(int), cudaMemcpyHostToDevice );
	// 
	// 	genAirFromSolid_k<<<gsblocknum, threadnum>>>( mParPos, mParVel, parflag, parsolubility, pargascontain, parmass, parTemperature, parNumNow,
	// 		mmark, phisolid, Tp, daddparnums, randfloat, randfloatcnt, mframe );
	// 
	// 	cudaMemcpy( &haddparnums, daddparnums, sizeof(int), cudaMemcpyDeviceToHost );
	// 	parNumNow += haddparnums;
	// 	pblocknum = max(1,(int)ceil(((float)parNumNow)/threadnum));
	// 	printf("add particle = %d\n", haddparnums );
}

void cspray::pouring()
{
	if (mframe % 4 == 0 && pourNum != 0 && pourNum + parNumNow <= parNumMax)
	{
		int tpblocknum = max(1, (int)ceil(((float)pourNum) / threadnum));
		//	printf("pournum=%d,parnumnow=%d,parnummax=%d,blocknum=%d", pourNum, parNumNow, parNumMax, tpblocknum );
		pouringwater << <tpblocknum, threadnum >> > (mParPos, mParVel, parmass, parflag, parTemperature, parLHeat, pargascontain, parNumNow,
			dpourpos, dpourvel, TYPEFLUID, pourNum, randfloat, randfloatcnt, 0, posrandparam, velrandparam, defaultLiquidT, LiquidHeatTh);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed");

		parNumNow += pourNum;
		pblocknum = max(1, (int)ceil(((float)parNumNow) / threadnum));
	}
}

void cspray::pouringgas()
{
	if (mframe % 6 == 0 && pourNum != 0 && pourNum + parNumNow <= parNumMax)
	{
		int tpblocknum = max(1, (int)ceil(((float)pourNum) / threadnum));
		//	printf("pournum=%d,parnumnow=%d,parnummax=%d,blocknum=%d", pourNum, parNumNow, parNumMax, tpblocknum );
		pouringwater << <tpblocknum, threadnum >> > (mParPos, mParVel, parmass, parflag, parTemperature, parLHeat, pargascontain, parNumNow,
			dpourpos, dpourvel, TYPEAIRSOLO, pourNum, randfloat, randfloatcnt, 0, posrandparam, velrandparam, defaultLiquidT, LiquidHeatTh);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed");

		parNumNow += pourNum;
		pblocknum = max(1, (int)ceil(((float)parNumNow) / threadnum));
	}
}

////liquid用SPH仿真，附近做一下流固耦合
void cspray::liquidUpdate_SPH()
{
	calcDensPressSPH_SLCouple << <pblocknum, threadnum >> >(mParPos, pardens, parpress, parflag, parNumNow, gridstart, gridend);
	cudaThreadSynchronize();

	//debug: check the density
	if (0)
	{
		static float *hdens = new float[parNumMax];
		static char *hflag = new char[parNumMax];
		cudaMemcpy(hdens, pardens, parNumNow*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(hflag, parflag, parNumNow*sizeof(char), cudaMemcpyDeviceToHost);

		float densaver = 0, densmax = -1, densmin = 10000000;
		int cnt = 0;
		for (int i = 0; i<parNumNow; i++)
		{
			densaver += 1 / hdens[i];
			densmax = max(densmax, 1 / hdens[i]);
			densmin = min(densmin, 1 / hdens[i]);
			cnt++;
		}
		densaver /= cnt;
		printf("densaver, max, min=%f,%f,%f\n", densaver, densmax, densmin);
	}

	enforceForceSPH_SLCouple << <pblocknum, threadnum >> >(mParPos, mParVel, pardens, parpress, parflag, parNumNow, gridstart, gridend, viscosiySPH);
	cudaThreadSynchronize();
}

void cspray::initHeatAlphaArray()
{
	float *halpha = new float[TYPECNT];
	halpha[TYPEFLUID] = heatalphafluid;
	halpha[TYPEAIR] = heatalphaair;
	halpha[TYPEVACUUM] = heatalphavacuum;
	halpha[TYPESOLID] = heatalphasolid;
	halpha[TYPEAIRSOLO] = heatalphaair;
	halpha[TYPEBOUNDARY] = 0;

	cudaMalloc((void**)&HeatAlphaArray, TYPECNT*sizeof(float));
	cudaMemcpy(HeatAlphaArray, halpha, TYPECNT*sizeof(float), cudaMemcpyHostToDevice);

	delete[] halpha;
}

