#include <stdio.h>
#include<cuda_runtime.h>
#include<helper_math.h>
#include "timer.h"
#include "spray.h"

//record the images.
#include "SOIL.h"

extern int winw, winh;


farray::farray()
{
	data = NULL; xn = NX; yn = NY; zn = NZ;
}

charray::charray()
{
	data = NULL; xn = NX; yn = NY; zn = NZ;
}

void cspray::initparam()
{
	//这里写经常需要改动的控制变量
	mscene = SCENE_ALL;
	readdata_solid();	//需要放在initparam之前，读取到固体点的个数

	mpause = false;
	boutputpovray = false;					//输出MC网格及表示spray的df3文件
	boutputobj = true;
	bOutputColoredParticle = true;
	outputframeDelta = 2;
	surfacetensionsigma = 1.0f;		//表面张力对于小气泡影响很大
	heatalphafluid = 0.02f, heatalphaair = 0.008f;
	heatalphasolid = 0.2f, heatalphavacuum = 0.0001f;
	defaulttemperature = 293.15f, Temperature0 = 273.15f;
	meltingpoint = 273.15f, boilingpoint = 373.15f;
	dragParamSolo = 10, dragParamGrid = 0.05f;
	viscosiySPH = 1.8f;
	pourNum = 0;
	m_bSolid = false;
	m_bMelt = false;
	m_bFreeze = false;
	m_bFixSolid = false;
	m_bGenGas = true;
	m_bHeatTrans = true;
	m_bAddHeatBottom = true;
	m_bExtendHeatToBoundary = false;
	m_bCorrectPosition = true;
	mRecordImage = false;
	SolidbounceParam = 0.5f;	//固体碰到边界时反弹的参数
	SolidBuoyanceParam = 0;	//0.05f; 浮力参数，但效果不好，考虑要不要计算流体的高度。
	defaultSolidT = 263.15f, defaultLiquidT = 293.15f, LiquidHeatTh = 10;
	posrandparam = 0.1f, velrandparam = 0.0f;
	alphaTempTrans = 0.0f;
	bounceVelParam = 1.0f, bouncePosParam = 0;
	seednum = 4;
	initdissolvegasrate = 1.0f, initgasrate = 0.7f;	//initial dissolved gas in liquid particle.
	vaporGenRate = 0.01f;
	temperatureMax_render = 373, temperatureMin_render = 273.0f;
	m_bSmoothMC = true;
	bRunMCSolid = true;
	m_DistanceFuncMC = 0;		//1表示使用2005年sand那篇论文的方法，0表示用高阳写的函数
	frameMax = 1500;
	m_bLiquidAndGas = false;
	m_bGas = true;				//false GY new； true YLP
	bubbleMaxVel = 6.5f;	//气泡上升的最大速度，气泡上升的速度越接近此值，浮力越小。限制气泡带起来的水花高度。
	m_bCPURun = false;
	updateSeedFrameDelta = 15;
	m_bCorrectFluidByAirLS = false;
	buoyanceRateAir = 1.25f, buoyanceRateSolo = 1.05f;
	m_beginFrame = 0;
	cntAirParMax = 0, cntLiquidMax = 0, cntSolidMax = 0;
	m_bBottomParticel = false;

	hparam.gravity = make_float3(0, 0, -9.8f);
	hparam.cellsize = make_float3(1.0f / 64);		//三个方向的步长相等，用1/64；这样粒子质量不随着场景变化，方便参数调整
	hparam.samplespace = 0.5f*hparam.cellsize.x;

	//sprintf(outputdir, "output\\povray\\");
	//params relate to scene
	if (mscene == SCENE_DAMBREAK)
	{
		bCouplingSphere = false;
		if (NX != 32 || NY != 32 || NZ != 32)
		{
			printf("reset NX,NY,NZ!!!\n");
			exit(-1000);
		}
		splitnum = 1;
		initfluidparticle = 81920;
		parNumNow = initfluidparticle + nInitSolPoint;
		parNumMax = parNumNow*splitnum;	//GY 流体粒子+固体粒子 //每个流体粒子都转为spray particle
		pourNum = 0;
		sprintf(outputdir, "output\\povraydambreak\\");
		simmode = SIMULATION_SOLIDCOUPLING;
		solidInitPos = make_float3(0.25f, 0.25f, 0.2f);				//固体初始位置
		m_bSolid = true;
	}
	else if (mscene == SCENE_FLUIDSPHERE)
	{
		if (NX != 24 || NY != 24 || NZ != 96) {
			printf("reset NX,NY,NZ!!!\n");
			exit(-1000);
		}
		splitnum = 2;
		initfluidparticle = 200000;
		parNumNow = initfluidparticle;
		parNumMax = parNumNow*splitnum;	//每个流体粒子都转为spray particle
		sprintf(outputdir, "output\\povrayfluidsphere\\");
		simmode = SIMULATION_WATER;
	}
	else if (mscene == SCENE_BOILING)
	{
		if (NX != 64 || NY != 64 || NZ != 64) {
			printf("reset NX,NY,NZ!!!\n");
			exit(-1000);
		}

		printf("%d,%d,%d\n", NX, NY, NZ);

		m_bSolid = false;
		m_bMelt = false;
		m_bFreeze = false;
		m_bGenGas = true;
		m_bHeatTrans = true;
		m_bAddHeatBottom = true;
		m_bExtendHeatToBoundary = true;

		splitnum = 2;
		initfluidparticle = 800000;
		parNumNow = initfluidparticle;
		parNumMax = parNumNow*splitnum;	//需要为气体粒子预留空间
		if (boutputpovray)
			sprintf(outputdir, "output\\povrayboiling\\");
		if (boutputobj)
			sprintf(outputdir, "outputobj\\objboiling\\");
		simmode = SIMULATION_BUBBLE;
		solidInitPos = make_float3(0.20f, 0.20f, 0.7f);
		defaulttemperature = 293.0f;
		vaporGenRate = 0.01f;
		defaultLiquidT = 293.0f;
		initdissolvegasrate = 0.95f, initgasrate = 0.88f;	//initial dissolved gas in liquid particle.
		heatIncreaseBottom = 1.5f;		//决定了底部热的增加速度
		heatalphafluid = 0.18f, heatalphaair = 0.08f;
		seednum = 2;
		dragParamSolo = 5, dragParamGrid = 0.12f;		//0.3 in TVCG 2014 paper, but it seems too high
		temperatureMax_render = 373, temperatureMin_render = 273.0f;
		frameMax = 1000;
	}
	else if (mscene == SCENE_BOILING_HIGHRES)
	{
		if (NX != 128 || NY != 128 || NZ != 96) {
			printf("reset NX,NY,NZ!!!\n");
			exit(-1000);
		}

		printf("%d,%d,%d\n", NX, NY, NZ);

		m_bSolid = false;
		m_bMelt = false;
		m_bFreeze = false;
		m_bGenGas = true;
		m_bHeatTrans = true;
		m_bAddHeatBottom = true;
		m_bExtendHeatToBoundary = true;

		//splitnum = 1.3;
		initfluidparticle = 3000000;
		parNumNow = initfluidparticle;
		parNumMax = 1000000 + parNumNow;	//需要为气体粒子预留空间
		if (boutputpovray)
			sprintf(outputdir, "output\\povrayboilingHighRes\\");
		if (boutputobj)
			sprintf(outputdir, "outputobj\\objboilingHighRes\\");
		simmode = SIMULATION_BUBBLE;
		solidInitPos = make_float3(0.20f, 0.20f, 0.7f);
		defaulttemperature = 293.0f;
		heatalphafluid = 0.2f, heatalphaair = 0.08f;
		vaporGenRate = 0.01f;
		defaultLiquidT = 293.0f;
		initdissolvegasrate = 0.95f, initgasrate = 0.88f;	//initial dissolved gas in liquid particle.
		heatIncreaseBottom = 3.0;// 1.5f;		//决定了底部热的增加速度
		heatalphafluid = 0.18f, heatalphaair = 0.08f;
		seednum = 3;//8;
		dragParamSolo = 5, dragParamGrid = 0.15f;		//  0.12    0.3 in TVCG 2014 paper, but it seems too high
		temperatureMax_render = 373, temperatureMin_render = 273.0f;
		frameMax = 2000;
	}
	else if (mscene == SCENE_MULTIBUBBLE)
	{ 
		m_bSolid = false;
		m_bMelt = false;
		m_bFreeze = false;
		m_bFixSolid = false;
		m_bGenGas = false;
		m_bHeatTrans = false;
		m_bAddHeatBottom = false;
		m_bExtendHeatToBoundary = false;
		m_bCorrectPosition = true;
		if (NX != 32 || NY != 32 || NZ != 64) {
			printf("reset NX,NY,NZ!!!\n");
			exit(-1000);
		}
		splitnum = 2;
		initfluidparticle = 250000;//460000;
		parNumNow = initfluidparticle;
		parNumMax = parNumNow*splitnum;	//每个流体粒子都转为spray particle
		sprintf(outputdir, "output\\povraymultibubble\\");
		simmode = SIMULATION_BUBBLE;
		solidInitPos = make_float3(0.5f, 0.5f, 0.2f);
		surfacetensionsigma = 0;	//关闭表面张力
	}
	else if (mscene == SCENE_MELTING)
	{
		bCouplingSphere = false;
		m_bMelt = true;
		m_bSolid = true;
		m_bAddHeatBottom = false;
		if (NX != 64 || NY != 64 || NZ != 64)
		{
			printf("reset NX,NY,NZ!!!\n");
			exit(-1000);
		}
		splitnum = 1;
		initfluidparticle = 350000;
		parNumNow = initfluidparticle + nInitSolPoint;
		parNumMax = parNumNow*splitnum;	//GY 流体粒子+固体粒子 //每个流体粒子都转为spray particle
		pourNum = 0;
		if (boutputpovray)
		sprintf(outputdir, "output\\povraymelting\\");
		if (boutputobj)
		sprintf(outputdir, "outputobj\\objmelting\\");
		simmode = SIMULATION_BUBBLE;
		solidInitPos = make_float3(0.5f, 0.5f, 0.15f);				//固体位置
		defaulttemperature = 293.15f;		//空气的温度定为20*C.
		heatalphafluid = 0.020000f;
		heatalphaair = 0.00008f;
	}
	else if (mscene == SCENE_MELTINGPOUR)
	{
		bCouplingSphere = false;
		m_bMelt = true;
		m_bSolid = true;
		m_bFixSolid = true;
		m_bFreeze = false;
		m_bGenGas = false;
		m_bAddHeatBottom = false;
		m_bExtendHeatToBoundary = true;
		bOutputColoredParticle = true;
		if (NX != 64 || NY != 64 || NZ != 64)
		{
			printf("reset NX,NY,NZ!!!\n");
			exit(-1000);
		}
		splitnum = 1;
		initfluidparticle = 0;
		parNumNow = initfluidparticle + nInitSolPoint;
		parNumMax = 100000 + parNumNow*splitnum;	//pouring + init
		pourNum = 0;
		pourRadius = 0.035f;
		pourpos = make_float3(0.6f, 0.45f, 0.75f);
		pourvel = make_float3(0.0f, 0.0f, -0.50f);
		CompPouringParam_Freezing();	//水龙头
		if (boutputpovray)
			sprintf(outputdir, "output\\povraymeltingpour\\");
		if (boutputobj)
			sprintf(outputdir, "outputobj\\objmeltingpour\\");
		simmode = SIMULATION_BUBBLE;
		solidInitPos = make_float3(0.55f, 0.45f, -0.33f);				//固体位置
		defaulttemperature = 279.15f;		//空气的温度
		meltingpoint = 268.15f;
		defaultSolidT = 260.15f, defaultLiquidT = 283.15f;
		heatalphafluid = 0.015f, heatalphaair = 0.025f;		//传热速度
		bounceVelParam = 1.3f, bouncePosParam = 1.0f;
		temperatureMax_render = 283, temperatureMin_render = 261;
		//alphaTempTrans = 1.0f;
	}
	else if (mscene == SCENE_FREEZING)
	{
		bCouplingSphere = false;
		m_bMelt = false;
		m_bSolid = true;
		m_bFixSolid = true;
		m_bFreeze = true;
		m_bCorrectPosition = true;
		m_bAddHeatBottom = true;
		bOutputColoredParticle = true;
		if (NX != 64 || NY != 64 || NZ != 64)
		{
			printf("reset NX,NY,NZ!!!\n");
			exit(-1000);
		}
		splitnum = 2;
		initfluidparticle = 0;
		parNumNow = initfluidparticle + nInitSolPoint;
		parNumMax = 100000 + parNumNow*splitnum;	//固体粒子+pouring的流体粒子
		pourNum = 0;
		pourRadius = 0.035f;
		pourpos = make_float3(0.60f, 0.48f, 0.75f);
		pourvel = make_float3(0.0f, 0.0f, -0.5f);
		CompPouringParam_Freezing();
		if (boutputpovray)
			sprintf(outputdir, "output\\povrayfreezing\\");
		if (boutputobj)
			sprintf(outputdir, "outputobj\\objfreezing\\");
		//simmode = SIMULATION_SOLIDCOUPLING;
		simmode = SIMULATION_BUBBLE;
		solidInitPos = make_float3(0.55f, 0.45f, -0.330f);				//固体位置
		defaulttemperature = 265.15f;
		heatalphafluid = 0.00001f, heatalphaair = 0.00008f;
		defaultSolidT = 263.15f, defaultLiquidT = 280.15f, LiquidHeatTh = 10;
		bounceVelParam = 1.0f, bouncePosParam = 0.5f;
		alphaTempTrans = 0.98f;
		viscosiySPH = 1.2f;
		//	hparam.gravity = make_float3(0);		//disable gravity.
		temperatureMax_render = 276, temperatureMin_render = 264;
	}
	else if (mscene == SCENE_INTERACTION)/////////
	{
		bCouplingSphere = false;
		m_bMelt = false;
		m_bFreeze = false;
		m_bFixSolid =true;
		m_bSolid = true;
		m_bGenGas = false;
		m_bHeatTrans = false;
		m_bAddHeatBottom = false;
		bRunMCSolid = false;
		if (NX != 64 || NY != 48 || NZ != 64)
		{
			printf("reset NX,NY,NZ!!!\n");
			exit(-1000);
		}
		splitnum = 1;
		initfluidparticle = 280000;
		parNumNow = initfluidparticle + nInitSolPoint;
		parNumMax = 100000 + parNumNow*splitnum;	//pouring gas particle.
		pourNum = 0;
		pourRadius = 0.05f;
		pourpos = make_float3(0.27f, 0.25f, 0.05f);
		pourpos2 = make_float3(0.45f, 0.25f, 0.05f);
		pourvel = make_float3(0.0f, 0.0f,1.1f);
		CompPouringParam_Ineraction2();
		if (boutputpovray)
		sprintf(outputdir, "output\\povrayinteraction\\");
		if (boutputobj)
		sprintf(outputdir, "outputobj\\objinteraction\\");
		simmode = SIMULATION_BUBBLE;
		solidInitPos = make_float3(0.41f, 0.28f, 0.2f);				//固体位置
		defaulttemperature = 293.15f;		//空气的温度定为20*C.
		heatalphafluid = 0.0008f, heatalphaair = 0.0008f;
		bounceVelParam = 1.3f, bouncePosParam = 1.0f;
		dragParamSolo = 5, dragParamGrid = 0.08f;
		surfacetensionsigma = 0.6f;
		m_beginFrame = 20;
	}
	else if (mscene == SCENE_INTERACTION_HIGHRES)
	{
		bCouplingSphere = false;
		m_bMelt = false;
		m_bFreeze = false;
		m_bFixSolid = false;
		m_bSolid = true;
		m_bGenGas = false;
		m_bHeatTrans = false;
		m_bAddHeatBottom = false;
		bRunMCSolid = false;
		if (NX != 96 || NY != 64 || NZ != 96)
		{
			printf("reset NX,NY,NZ!!!\n");
			exit(-1000);
		}
		splitnum = 1;
		initfluidparticle = 1500000;
		parNumNow = initfluidparticle + nInitSolPoint;
		parNumMax = 200000 + parNumNow*splitnum;	//pouring gas particle.
		pourNum = 0;
		pourRadius = 0.060f;	
		pourpos = make_float3(0.25f, 0.25f, 0.05f);
		pourpos = make_float3(0.25f, 0.25f, 0.05f);
		pourvel = make_float3(0.0f, 0.0f, 1.5f);
		CompPouringParam_Ineraction();
		if (boutputpovray)
			sprintf(outputdir, "output\\povrayinteraction_highres\\");
		if (boutputobj)
			sprintf(outputdir, "outputobj\\objinteraction_highres\\");
		simmode = SIMULATION_BUBBLE;
		solidInitPos = make_float3(0.750f, 0.58f, 0.8f);		//固体位置
		defaulttemperature = 293.15f;		//空气的温度定为20*C.
		heatalphafluid = 0.0008f, heatalphaair = 0.0008f;
		bounceVelParam = 1.3f, bouncePosParam = 1.0f;
		dragParamSolo = 5;//, dragParamGrid = 0.1f;
	}
	else if (mscene == SCENE_MELTANDBOIL)
	{
		bCouplingSphere = false;
		m_bMelt = true;
		m_bFreeze = false;
		m_bFixSolid = true;
		m_bSolid = true;
		m_bGenGas = true;
		m_bHeatTrans = true;
		m_bAddHeatBottom = true;
		m_bExtendHeatToBoundary = true;
		bRunMCSolid = true;
		if (NX != 64 || NY != 48 || NZ != 64)
		{
			printf("reset NX,NY,NZ!!!\n");
			exit(-1000);
		}

		splitnum = 1;
		initfluidparticle = 400000;
		parNumNow = initfluidparticle + nInitSolPoint;
		parNumMax = 100000 + parNumNow*splitnum;	//预留气泡粒子的空间
	    if (boutputpovray)
			sprintf(outputdir, "output\\povraymeltandboil\\");
		if (boutputobj)
			sprintf(outputdir, "outputobj\\objmeltandboil\\");
		simmode = SIMULATION_BUBBLE;
		solidInitPos = make_float3(0.50f, 0.38f, 0.2f);
		defaulttemperature = 293.0f;
		boilingpoint = 373.15f;
		// 		initgasrate = 0;		//dissolved gas is not included in this scene.
		// 		initdissolvegasrate=0;
		heatalphafluid = 0.1f, heatalphaair = 0.08f;		//决定温度的传递速度，这里比较大，因为需要水里很快的把温度传出去
		initdissolvegasrate = 0.95f, initgasrate = 0.95f;	//initial dissolved gas in liquid particle.
		defaultSolidT = 233.15f, defaultLiquidT = 350.15f, LiquidHeatTh = 20;
		vaporGenRate = 0.015f;		//决定了气体粒子生成的速率
		heatIncreaseBottom = 1.5f;		//决定了底部热的增加速度
		seednum = 2;
		temperatureMax_render = 373, temperatureMin_render = 353;
		dragParamSolo = 5, dragParamGrid = 0.03f;
		bubbleMaxVel = 9.5f;	//气泡上升的最大速度，气泡上升的速度越接近此值，浮力越小。限制气泡带起来的水花高度。
		updateSeedFrameDelta = 12;
		m_bCorrectFluidByAirLS = true;
		buoyanceRateAir = 1.45f, buoyanceRateSolo = 1.05f;
		
		frameMax = 2000;
	}
	else if (mscene == SCENE_MELTANDBOIL_HIGHRES)
	{
		bCouplingSphere = false;
		m_bMelt = true;
		m_bFreeze = false;
		m_bFixSolid = true;
		m_bSolid = true;
		m_bGenGas = true;
		m_bHeatTrans = true;
		m_bAddHeatBottom = true;
		m_bExtendHeatToBoundary = true;
		bRunMCSolid = true;
		if (NX != 96 || NY != 96 || NZ != 96)
		{
			printf("reset NX,NY,NZ!!!\n");
			exit(-1000);
		}

		splitnum = 1;
		initfluidparticle = 2400000;
		parNumNow = initfluidparticle + nInitSolPoint;
		parNumMax = 600000 + parNumNow*splitnum;	//预留气泡粒子的空间
		sprintf(outputdir, "output\\povraymeltandboilHighRes\\");
		simmode = SIMULATION_BUBBLE;
		solidInitPos = make_float3(0.850f, 0.68f, 0.65f);
		defaulttemperature = 293.0f;
		boilingpoint = 373.15f;
		heatalphafluid = 0.12f, heatalphaair = 0.08f;		//决定温度的传递速度，这里比较大，因为需要水里很快的把温度传出去
		initdissolvegasrate = 0.95f, initgasrate = 0.95f;	//initial dissolved gas in liquid particle.
		defaultSolidT = 243.15f, defaultLiquidT = 350.15f, LiquidHeatTh = 20;
		vaporGenRate = 0.017f;		//决定了气体粒子生成的速率
		heatIncreaseBottom = 1.5f;		//决定了底部热的增加速度
		seednum = 8;
		temperatureMax_render = 373, temperatureMin_render = 353;
		dragParamSolo = 5, dragParamGrid = 0.08f;
		bubbleMaxVel = 9.5f;	//气泡上升的最大速度，气泡上升的速度越接近此值，浮力越小。限制气泡带起来的水花高度。
		updateSeedFrameDelta = 12;
		m_bCorrectFluidByAirLS = true;
		buoyanceRateAir = 1.45f, buoyanceRateSolo = 1.05f;
		frameMax = 2000;
	}
	else if (mscene == SCENE_ALL)///////////////////ALLSET
	{
		bCouplingSphere = false;
		m_bMelt = true;
		m_bFreeze = false;
		m_bFixSolid = true;
		m_bSolid = true;
		m_bGenGas = true;
		m_bHeatTrans = true;
		m_bAddHeatBottom = true;
		m_bExtendHeatToBoundary = true;
		bRunMCSolid = true;
		if (NX !=96 || NY != 128 || NZ != 128)
		{
			printf("reset NX,NY,NZ!!!\n");
			exit(-1000);
		}

		splitnum = 1;
		initfluidparticle = 3100000;
		parNumNow = initfluidparticle + nInitSolPoint;
		parNumMax = 500000 + parNumNow*splitnum;	//预留气泡粒子的空间
		if (boutputpovray)
			sprintf(outputdir, "output\\povrayall\\");
		if (boutputobj)
			sprintf(outputdir, "outputobj\\objall\\");
		simmode = SIMULATION_BUBBLE;
		solidInitPos = make_float3(0.8f, 01.30f, 0.2f);
		defaulttemperature = 293.0f;
		boilingpoint = 373.15f;
		// 		initgasrate = 0;		//dissolved gas is not included in this scene.
		// 		initdissolvegasrate=0;
		heatalphafluid = 0.0001f, heatalphaair = 00.0008f;		//决定温度的传递速度，这里比较大，因为需要水里很快的把温度传出去
		initdissolvegasrate = 0.95f, initgasrate = 0.95f;	//initial dissolved gas in liquid particle.
		defaultSolidT = 223.15f, defaultLiquidT = 280.15f, LiquidHeatTh = 10;
		vaporGenRate = 0.015f;		//决定了气体粒子生成的速率
		heatIncreaseBottom = 50.0f;		//决定了底部热的增加速度
		seednum = 2;
		temperatureMax_render = 373, temperatureMin_render = 353;
		dragParamSolo = 5, dragParamGrid = 0.03f;
		bubbleMaxVel = 9.5f;	//气泡上升的最大速度，气泡上升的速度越接近此值，浮力越小。限制气泡带起来的水花高度。
		updateSeedFrameDelta = 12;
		m_bCorrectFluidByAirLS = true;
		buoyanceRateAir = 1.45f, buoyanceRateSolo = 1.05f;

		frameMax = 2000;
	}
	else if (mscene == SCENE_HEATTRANSFER)
	{
		if (NX != 256 || NY != 256 || NZ != 256)
		{
			printf("reset NX,NY,NZ!!!\n");
			exit(-1000);
		}
		simmode = SIMULATION_HEATONLY;
		m_bExtendHeatToBoundary = true;
		parNumNow = 0;
		parNumMax = 1;
		temperatureMax_render = 100.5f, temperatureMin_render = -0.5f;
		heatalphafluid = 0.0001f, heatalphaair = 0.005f;		//决定温度的传递速度，这里比较大，因为需要水里很快的把温度传出去
		heatalphasolid = 0.015f, heatalphavacuum = 0.05f;
		//special camera

	}
	initHeatAlphaArray();

	rendermode = RENDER_PARTICLE;
	colormode = COLOR_TP;
	velmode = FLIP;

	bsmoothMC = true;
	mframe = 0;
	bColorRadius = false;
	bColorVel = false;

	m_btimer = true;

	// 	dpourpos = NULL;
	// 	dpourvel = NULL;

	//default parameters.
	wind = make_float3(0.0f);
	bCouplingSphere = false;
	//default: for coupling with solid, simple code.
	sphereradius = 0.1f;
	bmovesphere = false;
	spherevel = make_float3(-1.0f, 0.0f, 0.0f);

	//param
	hparam.gnum = NX*NY*NZ;
	hparam.gvnum = make_int3((NX + 1)*NY*NZ, NX*(NY + 1)*NZ, NX*NY*(NZ + 1));
	hparam.dt = 0.003f;		//注意：pouring与时间步长有关
	hparam.gmin = make_float3(0.0f);
	hparam.gmax = hparam.cellsize.x * make_float3((float)NX, (float)NY, (float)NZ);		//x的长度总是1
	hparam.waterrho = 1000.0f;
	hparam.solidrho = 800.0f;
	hparam.m0 = (1.0f*hparam.waterrho) / pow(1.0f / hparam.samplespace, 3.0f);
	hparam.airm0 = hparam.m0;
	hparam.pradius = (float)(pow(hparam.m0 / hparam.waterrho*3.0f / 4.0f / M_PI, 1.0 / 3.0f));
	hparam.poly6kern = 315.0f / (64.0f * 3.141592f * pow(hparam.cellsize.x, 9.0f));
	hparam.spikykern = -45.0f / (3.141592f * pow(hparam.cellsize.x, 6.0f));
	hparam.lapkern = 45.0f / (3.141592f * pow(hparam.cellsize.x, 6.0f));

	maxVelForBubble = hparam.cellsize.x / hparam.dt;

	//MC
	NXMC = NX, NYMC = NY, NZMC = NZ;

	//time and parameters
	timeaver = new float[TIME_COUNT];
	timeframe = new float[TIME_COUNT];
	timemax = new float[TIME_COUNT];
	memset(timeaver, 0, sizeof(float)*TIME_COUNT);
	memset(timeframe, 0, sizeof(float)*TIME_COUNT);
	memset(timemax, 0, sizeof(float)*TIME_COUNT);

	//set cuda blocknum and threadnum.
	gsblocknum = (int)ceil(((float)NX*NY*NZ) / threadnum);
	int vnum = max((NX + 1)*NY*NZ, (NX)*(NY + 1)*NZ);
	vnum = max(vnum, NX*NY*(NZ + 1));
	gvblocknum = (int)ceil(((float)vnum) / threadnum);
	pblocknum = max(1, (int)ceil(((float)parNumNow) / threadnum));

	//MC parameters
	fMCDensity = 0.5f;
	smoothIterTimes = 5;

	velocitydissipation = 1.0f;
	densedissipation = 0.995f;
	fDenseDiffuse = 0.001f;
	fVelocityDiffuse = 0.001f;
	nDiffuseIters = 4;
	correctionspring = 500000.0;
	correctionradius = 0.5;

	//rand number
	randfloatcnt = 10000;
	renderpartiletype = TYPEAIR;

	

}

void cspray::init()
{
	initparam();
	initmem_bubble();
	printf("initmem complete.\n");
	initEmptyBubbles();		//决定emptybubble的数量，并分配存储

	initParticleGLBuffers();
	printf("initParticleGLBuffers complete.\n");

	initGridGLBuffers();
	printf("initGridGLBuffers complete.\n");
	initcubeGLBuffers();
	initdensityGLBuffers();

	//if (mscene == SCENE_FLUIDSPHERE)
	//	initscene_fluidsphere();
	//else if (mscene == SCENE_MULTIBUBBLE)
	//	initscene_bubble();
	//else
	if (mscene == SCENE_ALL)
		mmesh.LoadWithNor("objmodels/ss.obj");
	initparticle_solidCoupling();

	
	printf("initparticle complete.\n");

	initlight();
	loadshader();
	printf("loadshader complete.\n");

	initMC();
	printf("initMC complete.\n");

	initTemperature();
	initSolubility();
	initSeedCell();
	if (mscene == SCENE_HEATTRANSFER)
		initheat_grid();


	
}

void cspray::rollrendermode()
{
	rendermode = (ERENDERMODE)((rendermode + 1) % RENDER_CNT);
	printf("rendermode=%d\n", (int)rendermode);
}
void cspray::rollcolormode(int delta)
{
	colormode = (ECOLORMODE)((colormode + delta + COLOR_CNT) % COLOR_CNT);
	printf("colormode=%d\n", (int)colormode);
}

void printTime(bool btime, char* info, CTimer &time)
{
	if (!btime)
		return;
	double deltatime = time.stopgetstartMS();
	printf("%lf--", deltatime);
	printf(info);
	printf("\n");
}

void printTime(bool btime, char* info, CTimer &time, float* ptimeaver, float* ptimeframe, int timeflag)
{
	if (!btime)
		return;
	double deltatime = time.stopgetstartMS();
	//  	printf( "%lf--", deltatime);
	//  	printf( info );
	//  	printf("\n");

	//statistics
	ptimeaver[timeflag] += (float)deltatime;
	ptimeframe[timeflag] += (float)deltatime;
}

void cspray::simstep()
{
	if (simmode == SIMULATION_WATER) 
		watersim();
	else if (simmode == SIMULATION_SMOKE)
		smokesim();
	else if (simmode == SIMULATION_BUBBLE)//!
		bubblesim();
	else if (simmode == SIMULATION_SOLIDCOUPLING)
		SPHsimulate_SLCouple();
	else if (simmode == SIMULATION_HEATONLY)
		heatsim();

	//截图做视频
	static int frame = 0;
	if (!mpause && mRecordImage)
	{
		char str[100];
		//if( frame%2==0 )
		{
			sprintf(str, "output\\outputForHeatTransfer\\%05d.bmp", frame);
		//	SOIL_save_screenshot(str, SOIL_SAVE_TYPE_BMP, 0, 0, winw, winh);
		}
		frame++;
	}

	//统计时间
	if (mframe >= frameMax)
	{
		averagetime();
		exit(0);
	}
}

void cspray::watersim()
{
	if (!mpause)
	{
		CTimer time;
		time.startTimer();
		static CTimer timetotal;
		printTime(m_btimer, "TOTAL TIME!!", timetotal);

		if (m_btimer)
			printf("\n------------Frame %d:-------------\n", mframe);

		//		addexternalforce(); 
		printTime(m_btimer, "addexternalforce", time);

		hashAndSortParticles();
		printTime(m_btimer, "hashAndSortParticles", time);
		printTime(m_btimer, "markgrid", time);

		sweepPhi(phifluid, TYPEFLUID);
		printTime(m_btimer, "sweepPhi", time);

		computeLevelset(0);
		printTime(m_btimer, "computeLevelset", time);
		sweepLSAndMardGrid();
		flipMark_sphere();
		computesurfacetension();

		mapvelp2g();
		printTime(m_btimer, "mapvelp2g", time);

		sweepU(waterux, wateruy, wateruz, phifluid, mmark, TYPEFLUID);
		setWaterBoundaryU(waterux, wateruy, wateruz);
		printTime(m_btimer, "sweepU", time);

		project_CG(waterux, wateruy, wateruz);
		printTime(m_btimer, "project", time);

		//project之后sweepU的作用很大。可以设置一个场景，从底到上盛满了水，如果没有这一步，200帧左右就开始有些不正常了，400帧以后就非常明显了。
		sweepU(waterux, wateruy, wateruz, phifluid, mmark, TYPEFLUID);
		setWaterBoundaryU(waterux, wateruy, wateruz);
		printTime(m_btimer, "sweepU", time);

		mapvelg2p();
		printTime(m_btimer, "mapvelg2p", time);

		wateradvect();
		printTime(m_btimer, "advect", time);

		hashAndSortParticles();
		printTime(m_btimer, "hashAndSortParticles", time);
		correctpos();
		printTime(m_btimer, "cerrectpos", time);

		if (bsmoothMC)
			runMC_fluid();
		else
			runMC_flat(TYPEFLUID);
		printTime(m_btimer, "runMC", time);

		copyParticle2GL();

		mframe++;
	}
}

void cspray::waterSolidSim()
{
	if (!mpause)
	{
		CTimer time;
		time.startTimer();
		static CTimer timetotal;
		printTime(m_btimer, "TOTAL TIME!!", timetotal);

		if (m_btimer)
			printf("\n------------Frame %d:-------------\n", mframe);

		//if( mscene!=SCENE_MELTING )
		addexternalforce(); //计算的是v=v0-dt*g
		printTime(m_btimer, "addexternalforce", time);

		hashAndSortParticles();//建立哈希表 并分类粒子
		printTime(m_btimer, "hashAndSortParticles", time);

		mapvelp2g();		//粒子速度投影给网格
		printTime(m_btimer, "mapvelp2g", time);

		markgrid();			//	标记流体粒子网格和固体和边界网格
		printTime(m_btimer, "markgrid", time);

		sweepPhi(phifluid, TYPEFLUID);					//交换
		printTime(m_btimer, "sweepPhi", time);
		sweepU(waterux, wateruy, wateruz, phifluid, mmark, TYPEFLUID);							//交换速度			得到outu()
		setWaterBoundaryU(waterux, wateruy, wateruz);		//边界粒子速度为零
		printTime(m_btimer, "sweepU", time);

		project_CG(waterux, wateruy, wateruz);			//预处理得到ux uy uz
		printTime(m_btimer, "project", time);

		sweepU(waterux, wateruy, wateruz, phifluid, mmark, TYPEFLUID);							//交换速度			得到outu()
		setWaterBoundaryU(waterux, wateruy, wateruz);
		printTime(m_btimer, "sweepU", time);

		mapvelg2p();		//速度变化返回粒子
		printTime(m_btimer, "mapvelg2p", time);

		wateradvect();			//演进
		printTime(m_btimer, "advect", time);

		solidmotion();

		hashAndSortParticles();		////重新建立哈希表 并分类粒子
		correctpos();			//根据弹力修正粒子新位置
		printTime(m_btimer, "cerrectpos", time);

		hashAndSortParticles();
		if (m_bMelt)
		{
			updateTemperature();
			MeltSolid();
		}
		printTime(m_btimer, "updateHeat", time);

		runMC_fluid();

		printTime(m_btimer, "runMC", time);

		copyParticle2GL();

		mframe++;
	}
}

void cspray::smokesim()
{
	if (!mpause)
	{
		CTimer time;
		time.startTimer();
		static CTimer timetotal;
		printTime(m_btimer, "TOTAL TIME!!", timetotal);

		if (m_btimer)
			printf("\n------------Frame %d:-------------\n", mframe);

		smokesetvel();
		printTime(m_btimer, "smokesetvel", time);

		smokemarkgrid();
		printTime(m_btimer, "smokemarkgrid", time);

		setSmokeBoundaryU(msprayux, msprayuy, msprayuz);
		printTime(m_btimer, "setBoundaryU", time);

		project_Jacobi(msprayux, msprayuy, msprayuz);
		printTime(m_btimer, "project", time);

		setSmokeBoundaryU(msprayux, msprayuy, msprayuz);
		printTime(m_btimer, "setBoundaryU", time);

		smokeadvection();
		printTime(m_btimer, "smokeadvection", time);

		smokediffuse();
		printTime(m_btimer, "smokediffuse", time);

		copyDensity2GL();
		printTime(m_btimer, "copyDensity2GL", time);

		mframe++;
	}
}

//只关注liquid与solid，liquid用SPH仿真
void cspray::SPHsimulate_SLCouple()
{
	if (!mpause)
	{
		//mpause=true;
		CTimer time;
		time.startTimer();
		static CTimer timetotal;
		printTime(m_btimer, "TOTAL TIME!!", timetotal);

		if (m_btimer)
			printf("\n------------Frame %d:-------------\n", mframe);

		if (mscene == SCENE_INTERACTION || mscene == SCENE_INTERACTION_HIGHRES)
			pouringgas();
		else
			pouring();

		//更新seed positions.
		//if( mframe%15 == 0 )
		updateSeedCell();

		//1. external force: gravity, buoyancy, surface tension
		hashAndSortParticles();
		addexternalforce();
		printTime(m_btimer, "addexternalforce", time);

		//4. solid simulation step.
		if (m_bFixSolid)
			solidmotion_fixed();
		else
			solidmotion();

		//liquid update by SPH framework
		liquidUpdate_SPH();

		//7. update heat, generate/delete air/soloair particles; change the topology of solid.
		hashAndSortParticles();
		markgrid_bubble();
		if (m_bHeatTrans)
			updateTemperature();
		printTime(m_btimer, "updateHeat", time);

		//melting and freezing.
		if (m_bMelt)
			MeltSolid();
		if (m_bFreeze)
			Freezing();

		//8. rendering. notice: 必须进行hash (目前在mc算法的内部)，否则删除粒子之后就不正确了。
		runMC_solid();
		runMC_fluid();
		printTime(m_btimer, "runMC", time);

		copyParticle2GL();

		if (boutputpovray && mframe%outputframeDelta == 0)
			outputSoloBubblePovRay(mframe / outputframeDelta, mParPos, parmass, parflag, parNumNow);

		mframe++;
	}
}

void cspray::bubblesim()////////////////////////////
{
	if (!mpause)
	{
		//mpause=true;
		printf("Before MC: "), PrintMemInfo();

		CTimer time;
		CTimer time2;
		time2.startTimer();
		static CTimer timetotal;
		printTime(m_btimer, "TOTAL TIME!!", timetotal);

		memset(timeframe, 0, sizeof(float)*TIME_COUNT);

		if (m_btimer)
			printf("\n------------Frame %d:-------------\n", mframe);

		

		if ((mscene == SCENE_INTERACTION || mscene == SCENE_INTERACTION_HIGHRES) && mframe >= m_beginFrame)
			pouringgas();
		else
			pouring();

		//temp
		// 		if( mscene==SCENE_MELTANDBOIL && mframe==1 )
		// 			bRunMCSolid = true;
		if (mscene == SCENE_ALL & mframe == 0)
		{
			m_bFixSolid=false;
			m_bAddHeatBottom = false;
			m_bGenGas = false;
			
			m_bMelt = false;
		}
		if (mscene == SCENE_ALL && mframe == 0)
				{
				m_bFreeze = true;
			
				}
		if (mscene == SCENE_ALL && mframe == 500)
			m_bFreeze = false;
		if (mscene == SCENE_ALL && mframe == 520)
		{
			//	m_bFixSolid = true;
			m_bAddHeatBottom = true;
			m_bMelt = true;
		}
		
		if (mscene == SCENE_ALL && mframe == 1250)
		{
			m_bGenGas = true;
		}
		
		if ((mscene == SCENE_MELTANDBOIL || mscene == SCENE_MELTANDBOIL_HIGHRES) && mframe == 300)
			m_bFixSolid = false;
		if ((mscene == SCENE_MELTANDBOIL_HIGHRES ) && mframe == 400)
			m_bAddHeatBottom = false;
		if ((mscene == SCENE_MELTANDBOIL_HIGHRES) && mframe == 700)
			m_bAddHeatBottom = true;
		if ((mscene == SCENE_MELTANDBOIL) && mframe == 400)
			m_bGenGas = !m_bGenGas;
		if ((mscene == SCENE_MELTANDBOIL) && mframe == 700)
		{
			m_bGenGas = !m_bGenGas; dragParamGrid = 0.03f; vaporGenRate = 0.005f;
		}
		if ((mscene == SCENE_INTERACTION || mscene == SCENE_INTERACTION_HIGHRES) && mframe == m_beginFrame)
			m_bFixSolid = false;

		//更新seed positions.
		if (mframe > 800 && mscene == SCENE_BOILING)
			updateSeedFrameDelta = 10;
		if (mframe%updateSeedFrameDelta == 0)
			updateSeedCell();

		time.startTimer();

		//1. external force: gravity, buoyancy, surface tension
		hashAndSortParticles();
		addexternalforce();
		//printTime( m_btimer, "addexternalforce", time);
		printTime(m_btimer, "addexternalforce", time2);

		computeLevelset(0);	//todo: emptybubble
		sweepLSAndMardGrid();

		markSoloAirParticle();
		printTime(m_btimer, "computeLevelset", time2);

		//注意这里的表面张力模型，对于小气泡的影响很大。提交版本的论文里是没有这个的。
 		computesurfacetension();

		//利用粒子个数而不是level set来标记格子的性质，而不是level set
		if (/* mscene==SCENE_BOILING || */mscene == SCENE_MELTING || mscene == SCENE_FREEZING
			/*|| mscene==SCENE_MULTIBUBBLE*/ /* || mscene==SCENE_MELTANDBOIL*/ /*|| mscene==SCENE_MELTANDBOIL_HIGHRES*/)
			markgrid_bubble();
		if (mscene == SCENE_ALL)
			markgrid();
		printTime(m_btimer, "markgrid_bubble", time2);

		//3. grid-based solver
		mapvelp2g_bubble();
		printTime(m_btimer, "mapvelp2g", time2);

		//2. drag force between water and solo air particle
		enforceDragForce();
		printTime(m_btimer, "enforceDragForce", time2);

		sweepPhi(phiair, TYPEAIR);
		sweepU(airux, airuy, airuz, phiair, mmark, TYPEAIR);
		sweepPhi(phifluid, TYPEFLUID);
		sweepU(waterux, wateruy, wateruz, phifluid, mmark, TYPEFLUID);

		setWaterBoundaryU(airux, airuy, airuz);
		setWaterBoundaryU(waterux, wateruy, wateruz);
		printTime(m_btimer, "sweepU", time2);

		//空气和液体有两个速度场，统一计算压强并更新			//Section 3.2
		project_CG_bubble();
		printTime(m_btimer, "project_CG_bubble", time2);

		sweepU(airux, airuy, airuz, phiair, mmark, TYPEAIR);
		sweepU(waterux, wateruy, wateruz, phifluid, mmark, TYPEFLUID);
		setWaterBoundaryU(airux, airuy, airuz);
		setWaterBoundaryU(waterux, wateruy, wateruz);
		//printTime( mtime, "sweepU", time);
		printTime(m_btimer, "sweepU", time2);

		mapvelg2p_bubble();
		printTime(m_btimer, "mapvelg2p_bubble", time2);

		//4. solid simulation step.
		if (m_bFixSolid)
			solidmotion_fixed();
		else
			solidmotion();
		printTime(m_btimer, "solidmotion", time2);

		//4. SPH framework for solo air particle
		updateSoloAirParticle();
		printTime(m_btimer, "updateAirParticle", time2);

		//5. advect fluid/air/airsolo/solid particle
	//	advect_bubble();
		//printTime( m_btimer, "advect", time);
		printTime(m_btimer, "advect_bubble", time2);

		hashAndSortParticles();
		CollisionSolid();
		printTime(m_btimer, "CollisionSolid", time2);

		//6. correct all positions, distribute evenly.
		if (m_bCorrectPosition)
		{
			hashAndSortParticles();
			correctpos();			//todo: emptybubble
			//printTime( m_btimer, "correctpos", time);

			//correctpos_bubble是一个关键，如果空气的粒子过于聚合，之后乱飞，大半是这里的问题。
			hashAndSortParticles();
			computeLevelset(0.15f);
			sweepLSAndMardGrid();
			correctpos_bubble();
			//printTime( m_btimer, "correctpos_bubble", time);
		}
		else
		{
			hashAndSortParticles();
			computeLevelset(0.15f);
			sweepLSAndMardGrid();
		}
		printTime(m_btimer, "dynamics", time, timeaver, timeframe, TIME_DYNAMICS);
		printTime(m_btimer, "correctpos", time2);

		//7. update heat, generate/delete air/soloair particles; change the topology of solid.
		hashAndSortParticles();
		//		markgrid_bubble();
		if (m_bHeatTrans)
			updateTemperature();
		printTime(m_btimer, "updateTemperature", time2);

		int genGasFrameDelta = 2;
		// 		if(mframe>300) genGasFrameDelta = 3;
		// 		if(mframe>450) genGasFrameDelta = 4;
		if (m_bGenGas && mframe%genGasFrameDelta == 0)
			GenerateGasParticle();		//
		printTime(m_btimer, "GenerateGasParticle", time2);

		hashAndSortParticles();
		deleteAirFluidParticle();
		
		printTime(m_btimer, "deleteAirParticle", time2);

		//melting and freezing.
		hashAndSortParticles();
		printTime(m_btimer, "hashAndSortParticles", time2);
		if (m_bMelt)
		{
			if (m_bCPURun)
				MeltSolid_CPU();
			else
				MeltSolid();
		}
		if (m_bFreeze)
			Freezing();
		printTime(m_btimer, "MeltSolid&Freezing", time2);

		printTime(m_btimer, "dynamics", time, timeaver, timeframe, TIME_TRANSITION);

		//8. rendering. notice: 必须进行hash，否则删除粒子之后就不正确了。
		if (mframe%outputframeDelta == 0)
		{
			preMC();
			if (mscene == SCENE_INTERACTION || mscene == SCENE_INTERACTION_HIGHRES || mscene == SCENE_MELTANDBOIL || mscene == SCENE_MELTANDBOIL_HIGHRES ||mscene ==SCENE_ALL)
				runMC_interaction();
			else
				runMC_solid();
			runMC_fluid();
			runMC_gas();
		}
		//printTime( m_btimer, "runMC", time);
		copyParticle2GL();

		printTime(m_btimer, "dynamics", time, timeaver, timeframe, TIME_DISPLAY);
		printTime(m_btimer, "MC", time2);

		if (boutputpovray && mframe%outputframeDelta == 0)
		{
			if (mscene == SCENE_INTERACTION)
				outputSoloBubblePovRay(mframe / outputframeDelta, mParPos, parmass, parflag, parNumNow);
			if (bOutputColoredParticle)
				outputColoredParticle(mframe / outputframeDelta, mParPos, parTemperature, parNumNow);
			if (mscene == SCENE_BOILING)
				outputAirParticlePovRay(mframe / outputframeDelta, mParPos, parmass, parflag, parNumNow);
		}
		printTime(m_btimer, "output", time2);

		//time statistics;
		//处理一下时间，每一帧的最大时间
		for (int i = 0; i < TIME_COUNT; i++)
		{
			if (timemax[i] < timeframe[i])
				timemax[i] = timeframe[i];
			printf("timemax %d: %f\n", i, timeframe[i]);
		}

		if (mframe % 5 == 0)
			statisticParticleflag(mframe, parflag, parNumNow);

		mframe++;
	}
}

void cspray::resetsim()
{
	parNumNow = 0;
	int gsmemsize = sizeof(float)*hparam.gnum;
	cudaMemset(mpress.data, 0, gsmemsize);
	cudaMemset(temppress.data, 0, gsmemsize);
	cudaMemset(mDiv.data, 0, gsmemsize);
	cudaMemset(phifluid.data, 0, gsmemsize);
	cudaMemset(phiair.data, 0, gsmemsize);

	//u
	int gvxmemsize = sizeof(float)*hparam.gvnum.x;
	int gvymemsize = sizeof(float)*hparam.gvnum.y;
	int gvzmemsize = sizeof(float)*hparam.gvnum.z;

	cudaMemset(waterux.data, 0, gvxmemsize);
	cudaMemset(wateruy.data, 0, gvymemsize);
	cudaMemset(wateruz.data, 0, gvzmemsize);
	cudaMemset(waterux_old.data, 0, gvxmemsize);
	cudaMemset(wateruy_old.data, 0, gvymemsize);
	cudaMemset(wateruz_old.data, 0, gvzmemsize);

	cudaMemset(mParVel, 0, parNumMax*sizeof(float3));

	cudaMemset(spraydense.data, 0, sizeof(float)*hparam.gnum);
	cudaMemset(msprayux.data, 0, gvxmemsize);
	cudaMemset(msprayuy.data, 0, gvymemsize);
	cudaMemset(msprayuz.data, 0, gvzmemsize);
}

void cspray::averagetime()
{
	for (int i = 0; i < TIME_COUNT; i++)
		timeaver[i] /= mframe;
	//输出
	char str[100];
	sprintf(str, "statistictime%d.txt", (int)(mscene));
	FILE* fp = fopen(str, "w");
	fprintf(fp, "max liquid particle=%d, max gas particle=%d, max solid particle=%d\n\n", cntLiquidMax, cntAirParMax, cntSolidMax);
	fprintf(fp, "time:\n");
	for (int i = 0; i < TIME_COUNT; i++)
	{
		fprintf(fp, "time_index=%d, timeaver=%.2f, timemax=%.2f\n", i, timeaver[i], timemax[i]);
	}
	fclose(fp);
}

void cspray::movesphere()
{
	if (!bmovesphere)
		return;
	solidInitPos += spherevel * hparam.dt;
	if (solidInitPos.x < 0.5f)
	{
		solidInitPos.x = 0.5f;
		spherevel.x = abs(spherevel.x);
	}
	if (solidInitPos.x>0.9f)
	{
		solidInitPos.x = 0.9f;
		spherevel.x = -abs(spherevel.x);
	}
}

float3 cspray::mapColorBlue2Red_h(float v)
{
	float3 color;
	if (v < 0)
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

void cspray::statisticParticleflag(int frame, char *dflag, int pnum)
{
	static char *hflag = new char[parNumMax];
	cudaMemcpy(hflag, dflag, pnum*sizeof(char), cudaMemcpyDeviceToHost);
	int cntAirPar = 0, cntLiquid = 0, cntSolid = 0;
	for (int i = 0; i < pnum; i++)
	{
		if (hflag[i] == TYPEAIRSOLO || hflag[i] == TYPEAIR)
			cntAirPar++;
		else if (hflag[i] == TYPEFLUID)
			cntLiquid++;
		else if (hflag[i] == TYPESOLID)
			cntSolid++;
	}
	cntLiquidMax = max(cntLiquid, cntLiquidMax);
	cntAirParMax = max(cntAirPar, cntAirParMax);
	cntSolidMax = max(cntSolid, cntSolidMax);

	//printf( "droplet particle: %d, liquid particle: %d\n", cntAirPar, cntLiquid );
}

class MyMatrix33
{
public:
	MyMatrix33(
		float a00, float a01, float a02,
		float a10, float a11, float a12,
		float a20, float a21, float a22)
	{
		data[0][0] = a00;
		data[0][1] = a01;
		data[0][2] = a02;
		data[1][0] = a10;
		data[1][1] = a11;
		data[1][2] = a12;
		data[2][0] = a20;
		data[2][1] = a21;
		data[2][2] = a22;
	}

	float3 Multiple(const float3 &v)
	{
		float3 result;
		result.x = data[0][0] * v.x + data[1][0] * v.y + data[2][0] * v.z;
		result.y = data[0][1] * v.x + data[1][1] * v.y + data[2][1] * v.z;
		result.z = data[0][2] * v.x + data[1][2] * v.y + data[2][2] * v.z;
		return result;
	}

private:
	float data[3][3];
};

void cspray::CompPouringParam_Freezing()
{
	//
	int num = (int)(pourRadius / hparam.samplespace)*3;

	int memnum = (2 * num + 1)*(2 * num + 1) ;
	float3 *hpourpos = new float3[memnum];
	float3 *hpourvel = new float3[memnum];
	int cnt = 0;
	float3 ipos;
	for (int x = -num; x <= num; x++) for (int y = -num; y <= num; y++)
	{
		ipos = pourpos + make_float3(x*hparam.samplespace, y*hparam.samplespace, 0);
		if (length(ipos - pourpos) > pourRadius)
			continue;
		hpourpos[cnt] = ipos;
		hpourvel[cnt] = pourvel;

		cnt++;
	}
	if (cnt == 0)
		printf("pouring num=0!!!\n");
	else
	{
		//直接pour两层粒子
		//for( int i=0; i<cnt; i++ )
		//{
		//	hpourpos[cnt+i] = hpourpos[i] + 0.5f*hparam.dt*pourvel;
		//	hpourvel[cnt+i] = hpourvel[i];
		//}
		//cnt*=2;

		printf("pouring num=%d\n", cnt);
		
		cudaMalloc((void**)&dpourpos, sizeof(float3)*cnt);
		cudaMalloc((void**)&dpourvel, sizeof(float3)*cnt);
		cudaMemcpy(dpourpos, hpourpos, sizeof(float3)*cnt, cudaMemcpyHostToDevice);
		cudaMemcpy(dpourvel, hpourvel, sizeof(float3)*cnt, cudaMemcpyHostToDevice);
	}
	pourNum = cnt;
	
	delete[] hpourpos;
	delete[] hpourvel;
}

void cspray::CompPouringParam_Ineraction()
{
	//
	float ss = hparam.samplespace * 2.0f;
	int num = (int)(pourRadius / ss);
	int memnum = (2 * num + 1)*(2 * num + 1) * 2;
	float3 *hpourpos = new float3[memnum];
	float3 *hpourvel = new float3[memnum];
	int cnt = 0;
	float3 ipos;
	for (int x = -num; x <= num; x++) for (int y = -num; y <= num; y++)
	{
		ipos = pourpos + make_float3(x*ss, y*ss, 0);
		if (length(ipos - pourpos) > pourRadius)
			continue;
		hpourpos[cnt] = ipos;
		hpourvel[cnt] = pourvel;

		cnt++;
	}
	if (cnt == 0)
		printf("pouring num=0!!!\n");
	else
	{
		printf("pouring num=%d\n", cnt);
		cudaMalloc((void**)&dpourpos, sizeof(float3)*cnt);
		cudaMalloc((void**)&dpourvel, sizeof(float3)*cnt);
		cudaMemcpy(dpourpos, hpourpos, sizeof(float3)*cnt, cudaMemcpyHostToDevice);
		cudaMemcpy(dpourvel, hpourvel, sizeof(float3)*cnt, cudaMemcpyHostToDevice);
	}
	pourNum = cnt;

	delete[] hpourpos;
	delete[] hpourvel;
}

void cspray::CompPouringParam_Ineraction2()
{
	//
	float ss = hparam.samplespace * 2.0f;
	int num = (int)(pourRadius / ss);
	int memnum = (2 * num + 1)*(2 * num + 1) * 4;
	float3 *hpourpos = new float3[memnum];
	float3 *hpourvel = new float3[memnum];
	int cnt = 0;
	float3 ipos;
	for (int x = -num; x <= num; x++) for (int y = -num; y <= num; y++)
	{
		ipos = pourpos + make_float3(x*ss, y*ss, 0);
		if (length(ipos - pourpos) > pourRadius)
			continue;
		hpourpos[cnt] = ipos;
		hpourvel[cnt] = pourvel;

		cnt++;
	}

	for (int x = -num; x <= num; x++) for (int y = -num; y <= num; y++)
	{
		ipos = pourpos2 + make_float3(x*ss, y*ss, 0);
		if (length(ipos - pourpos2) > pourRadius)
			continue;
		hpourpos[cnt] = ipos;
		hpourvel[cnt] = pourvel;

		cnt++;
	}

	if (cnt == 0)
		printf("pouring num=0!!!\n");
	else
	{
		printf("pouring num=%d\n", cnt);
		cudaMalloc((void**)&dpourpos, sizeof(float3)*cnt);
		cudaMalloc((void**)&dpourvel, sizeof(float3)*cnt);
		cudaMemcpy(dpourpos, hpourpos, sizeof(float3)*cnt, cudaMemcpyHostToDevice);
		cudaMemcpy(dpourvel, hpourvel, sizeof(float3)*cnt, cudaMemcpyHostToDevice);
	}
	pourNum = cnt;

	delete[] hpourpos;
	delete[] hpourvel;
}
void cspray::markgird_terrain()
{
	charray hmark;
	hmark.data = new char[hparam.gnum];
	

	float height[NX + 1][NY + 1];
	for (int i = 0; i < mmesh.m_nPoints; i++)
	{
		int x = (int)floor(mmesh.m_hPoints[i].x / hparam.cellsize.x + 0.5);	//scale到0-NX
		int y = (int)floor(mmesh.m_hPoints[i].y / hparam.cellsize.x + 0.5);	//scale到0-NY
		height[x][y] = mmesh.m_hPoints[i].z;
	}

	//x与y上进行遍历
 	for (int x = 0; x < NX; x++) for (int y = 0; y < NY; y++)
 	{
 		//四个角高度平均
 		float h = (height[x][y] + height[x + 1][y] + height[x][y + 1] + height[x + 1][y + 1])*0.25f;
 		int hmax = (int)floor(h / hparam.cellsize.x + 0.5f);
 		for (int z = 0; z < NZ; z++)
 		{
 			if (z < hmax)
 				hmark(x, y, z) = TYPEBOUNDARY;
			else
 				hmark(x, y, z) = TYPEAIR;
 		}
 	}

	cudaMemcpy(mark_terrain.data, hmark.data, hparam.gnum*sizeof(char), cudaMemcpyHostToDevice);

	int3 minbottom = make_int3(1, 1, 1), maxbottom = make_int3(NX - 2, NY - 2, 6);// 底部水的高度
	initBottomParticles_terrain(minbottom, maxbottom, height);
	
}
void cspray::initBottomParticles_terrain(int3 mincell, int3 maxcell, float height[NX + 1][NY + 1])//////////////
{
	float ss = hparam.samplespace;
	float3 minpos = make_float3((float)mincell.x, (float)mincell.y, (float)mincell.z) * hparam.cellsize.x + hparam.samplespace;
	float3 maxpos = make_float3(maxcell.x + 1.0f, maxcell.y + 1.0f, maxcell.z + 1.0f) * hparam.cellsize.x;          
	//float3 maxpos = make_float3( NX-1,NY-1,(float)Waterhight+1 ) * hparam.cellsize.x;
	float3 delta = (maxpos - minpos) / hparam.samplespace;
	int num = ((int)(delta.x + 1)) * ((int)(delta.y + 1)) * ((int)(delta.z + 1));
	parNumNow += num; //流体+固体
	float x, y, z;
	float3 *hparpos = new float3[parNumNow];
	float3* hparvel = new float3[parNumNow];
	float* hparmass = new float[parNumNow];
	char* hparflag = new char[parNumNow];
	int cnt = 0;
	float dx = hparam.cellsize.x;

	float scale = 50;
	if (mscene == SCENE_FREEZING || mscene == SCENE_MELTINGPOUR) scale = 80;
	if (mscene == SCENE_INTERACTION) scale = 60;
	if (mscene == SCENE_MELTANDBOIL_HIGHRES || mscene == SCENE_INTERACTION_HIGHRES) scale = 100;
	if (mscene == SCENE_ALL) scale = 80;

	for (int j = 0; j < nInitSolPoint; j++)
	{
		x = float(SolpointPos[j][0]), y = float(SolpointPos[j][1]), z = float(SolpointPos[j][2]);
		hparpos[cnt] = hparam.samplespace*make_float3(x, y, z)*scale + solidInitPos;
		hparvel[cnt] = make_float3(0.0f);		//	
		hparmass[cnt] = hparam.m0*0.8f;
		hparflag[cnt] = TYPESOLID;	//类型是固体
		cnt++;
	}

	for (float x = minpos.x; x <= maxpos.x; x += ss) for (float y = minpos.y; y <= maxpos.y; y += ss) for (float z = minpos.z; z <= maxpos.z; z += ss)
	{
		//地形检测
		int i = (int)floor(x / hparam.cellsize.x);
		int j = (int)floor(y / hparam.cellsize.x);
		float h = (1 - y / dx + j) * ((1 - x / dx + i)*height[i][j] + (x / dx - i)*height[i + 1][j]) + (y / dx - j) * ((1 - x / dx + i)*height[i][j + 1] + (x / dx - i) *height[i + 1][j + 1]);
		if (z < h)
			continue;

		hparpos[cnt] = make_float3(x, y, z);
		hparvel[cnt] = make_float3(0.0f);
		hparmass[cnt] = hparam.m0;
		hparflag[cnt] = TYPEFLUID;
		cnt++;
	}
	


	printf("initParticlesnow num=%d\n", cnt);
	if (cnt != 0)
	{
		cudaMemcpy(mParPos, hparpos, sizeof(float3)*cnt, cudaMemcpyHostToDevice);
		cudaMemcpy(mParVel, hparvel, sizeof(float3)*cnt, cudaMemcpyHostToDevice);
		cudaMemcpy(parmass, hparmass, sizeof(float)*cnt, cudaMemcpyHostToDevice);
		cudaMemcpy(parflag, hparflag, sizeof(char)*cnt, cudaMemcpyHostToDevice);

	}
//	waterNumNow = cnt;
	
	parNumNow = cnt;
	
	pblocknum = max(1, (int)ceil(((float)parNumNow) / threadnum));

	delete[] hparpos;
	delete[] hparvel;
	delete[] hparmass;
	delete[] hparflag;
}
