#ifndef SPRAY_H
#define SPRAY_H

#include<vector_types.h>
#include "utility.h"
#include "mymesh.h"

//NX,NY,NZ用来控制仿真空间在三个方向上分别有几个cell
//const int NX=256,NY=4,NZ=256;
//const int NX=48,NY=48,NZ=48;  
//const int NX=128,NY=128,NZ=128;	//meltingpour / 
//const int NX = 128, NY = 128, NZ = 96;  //boiling high resolution
//const int NX=96,NY=128,NZ=128;
const int NX=64	,NY=64,NZ=64;//freezing
//const int NX=48,NY=32,NZ=64;  //interaction
//const int NX=64,NY=48,NZ=64; //boiling / melting

class cspray
{
public:
	void init();
	void initparam();
	void simstep();
	void project_CG(farray ux, farray uy, farray uz);
	void project_Jacobi(farray ux, farray uy, farray uz);
	void wateradvect();
	void correctpos();
	void render();
	void mapvelp2g();
	void mapvelg2p();
	void addexternalforce();
	void watersim();
	void initParticleGLBuffers();
	void initGridGLBuffers();
	void rendergridvariables();
	void renderparticleshader();
	void loadshader();
	void markgrid();
	void setGridColor();
	void sweepPhi(farray phi, char typeflag);
	void sweepU(farray ux, farray uy, farray uz, farray phi, charray mark, char typeflag);
	void setWaterBoundaryU(farray ux, farray uy, farray uz);
	void setSmokeBoundaryU(farray ux, farray uy, farray uz);
	void hashAndSortParticles();
	void hashAndSortParticles_MC();
	void copyParticle2GL();
	void initlight();

	//只关注liquid与solid，liquid用SPH仿真
	void SPHsimulate_SLCouple();
	void liquidUpdate_SPH();

	//coupling with solid. simple code.
	void rendersphere();
	void renderbox_waterfall(int3 minpos, int3 maxpos);
	void movesphere();

	float3 solidInitPos;
	float sphereradius;
	float3 spherevel;
	bool bmovesphere;
	int3 waterfallminpos, waterfallmaxpos;
	char outputdir[30];
	//for spray&smoke simulation.
	//void smokeinit();
	void smokeadvection();
	void smokesim();
	void smokemarkgrid();
	void markgrid_bubble();
	void smokesetvel();
	void smokeRayCasting();
	void initcubeGLBuffers();
	void initdensityGLBuffers();
	void copyDensity2GL();
	void checkdensesum();
	void smokediffuse();
	farray spraydense, tmpspraydense;
	farray msprayux, msprayuy, msprayuz;
	farray mtmpsprayux, mtmpsprayuy, mtmpsprayuz;

	void resetsim();

	bool solver_cg(charray A, farray x, farray b, int n);
	void solver_Jacobi(charray A, farray x, farray b, int itertime);
	float product(farray a, farray b, int n);
	uint splitnum;
	//for PCG
	farray pre, z, r, p;

	//for pouring water
	int pourNum;		//一次发射粒子个数，最好是一个平方数
	float3 pourvel, pourpos, pourpos2;
	float3 *dpourvel, *dpourpos;
	float posrandparam, velrandparam;
	float pourRadius;
	int pourDirType;	//大致表示方向，例如：向x方向发射，则yz坐标做一个表面，pourDirType=1；y: =2; z: =3

	//for wind
	float3 wind;

	//marching cube.
	void initMC();
	void runMC_flat(char MCParType);
	void runMC_smooth(const char* objectname, char MCParType);
	void initMCtriangles(int _maxVerts, int _maxTriangles);
	void renderIsosurface_flat();	//render each vertex, smooth rendering
	void renderIsosurface_smooth();	//render each triangle, smooth rendering
	void calNormals(float3 *dnormals, float3 *dpos, int vertexnum, uint *dindices, int indicesnum);
	void smoothMesh(float3 *dpos, int vertexnum, uint *indices, int trianglenum);
	void runMC_fluid();
	void runMC_solid();
	void runMC_gas();
	void runMC_interaction();
	void preMC();

	bool m_bLiquidAndGas, m_bGas;
	farray waterdensMC;
	unsigned int posVbo, normalVbo, indicesVbo;
	struct cudaGraphicsResource *res_posvbo, *res_normvbo, *res_indicesvbo; // handles OpenGL-CUDA exchange
	uint activeVoxels, totalVerts, totalIndices;
	float fMCDensity;		//控制MC算法的精度
	uint *d_voxelVerts;
	uint *d_voxelVertsScan;
	uint *d_voxelOccupied;
	uint *d_voxelOccupiedScan;
	uint *d_compVoxelArray;
	uint numVoxels, maxVerts, maxTriangles, MCedgeNum;
	uint *MCedgemark, *MCedgemarkScan;
	//for smoothing the triangles in 3d mesh created by MC
	float3 *smoothdisplacement;
	int *smoothweight;
	int smoothIterTimes;
	bool m_bSmoothMC;		//是否smooth MC算法生成的mesh
	int m_DistanceFuncMC;	//0表示用高阳写的genWaterDensfield_GY函数，1表示用原来的genWaterDensfield2函数
	float3 *solidvertex, *solidnormal;
	int solidvertexnum, solidindicesnum;
	uint *solidindices;
	bool bRunMCSolid;

	//help function.
	void swapParticlePointers();
	void PrintMemInfo();
	void rollrendermode();
	void rollcolormode(int delta);
	int getblocknum(int n);
	void checkparticlevariables(float3* dvel);
	float checkGridFarray(farray u);
	float3 mapColorBlue2Red_h(float v);
	void averagetime();
	void statisticParticleflag(int frame, char *dflag, int pnum);
	int cntAirParMax, cntLiquidMax, cntSolidMax;

	FlipConstant hparam;
	static const int threadnum = 512;
	int gsblocknum, gvblocknum, pblocknum;		//CUDA block个数：标量场、向量场、粒子

	//data for eulerian feild of liquid.
	farray mDiv, waterux, wateruy, wateruz, phifluid, phiair;
	farray waterux_old, wateruy_old, wateruz_old;	//注意，这里存的是project之前的速度，不是上一帧的速度
	farray tmpux, tmpuy, tmpuz;
	farray mpress, temppress;
	charray mmark, mark_terrain;

	//for FLIP particle ( liquid particle )
	float3 *mParPos, *mParVel, *tmpParPos, *tmpParVelFLIP;
	char *parflag, *tmpparflag;
	float *parmass, *tmpparmass;
	float *parTemperature, *tmpparTemperature;
	float *parLHeat, *tmpparHeat;		//latent heat for particle.
	int parNumNow, parNumMax;	//meaning: present particle number(water and spray particles), max particle number.
	int initfluidparticle;


	//cell for particles sort.
	uint *gridHash, *gridIndex;
	uint *gridstart, *gridend;

	//rendering:
	unsigned int mProgramParticle, mProgramGrid, mProgramSmoke;
	unsigned int vaoPar, vboParPos, vboParColor, vaoGrid, vboGridpos, vboGridcolor;
	unsigned int vaocube, vbocube;
	unsigned int densTex3DID;
	struct cudaGraphicsResource *densTex3D_cuda; // CUDA Graphics Resource (to transfer PBO)

	//control 
	bool mpause, m_btimer;
	ERENDERMODE rendermode;
	ECOLORMODE colormode;	//用什么模式来显示网格上的参数
	VELOCITYMODEL velmode;
	SIMULATIONMODE simmode;
	SCENE mscene;
	bool bsmoothMC;
	int mframe, frameMax;
	bool bCouplingSphere;
	bool bColorRadius, bColorVel;

	//rand number for CUDA
	float *randfloat;
	int randfloatcnt;

	//scene related parameters.
	float velocitydissipation, densedissipation;
	float fDenseDiffuse, fVelocityDiffuse;
	int nDiffuseIters;
	float correctionspring, correctionradius;

	//statistics time and particle number.
	float *timeaver;
	float *timeframe;
	float *timemax;

	//terrain相关的函数
	myMesh mmesh;
	void markgird_terrain();
	void initBottomParticles_terrain(int3 mincell, int3 maxcell, float height[NX + 1][NY + 1]);

	void ComputeTriangleHashSize(myMesh &mesh);
	void hashTriangle_radix_q();
	void sortTriangles_q(uint numParticles);
	void reorderTriangle_radix_q();
	void updateNormal_q();
	void Coupling_f2s_q(int scene);

	//bubble simulation
	void bubblesim();
	void initscene_bubble();
	void initmem_bubble();
	void computeLevelset(float offset);
	void mapvelp2g_bubble();
	void project_CG_bubble();
	bool solver_cg_bubble(charray A, farray x, farray b, int n);
	void advect_bubble();
	void mapvelg2p_bubble();
	bool solver_cg_heat(charray A, farray x, farray b, int n);
	void initTemperature();
	void updateTemperature();
	void computesurfacetension();
	void sweepLSAndMardGrid();
	void correctpos_bubble();
	void deleteAirFluidParticle();
	void renderIsosurface_transparent();
	void initscene_multibubble();
	void updateSoloAirParticle();
	void markSoloAirParticle();
	void initSolubility();
	void GenerateGasParticle();
	void MeltSolid();
	void MeltSolid_CPU();
	void enforceDragForce();
	void updateLatentHeat();
	void Freezing();

	void heatsim();
	void initheat_grid();
	void initHeatAlphaArray();

	void pouring();
	void pouringgas();
	void CompPouringParam_Freezing();
	void CompPouringParam_Ineraction();
	void CompPouringParam_Ineraction2();

	//test.
	void initscene_fluidsphere();
	void flipMark_sphere();

	farray lsair, lsfluid, lsmerge;	//level set
	farray airux, airuy, airuz, airux_old, airuy_old, airuz_old;
	farray Tp, Tp_old;
	farray Tp_save;
	farray surfacetension;
	farray phigrax, phigray, phigraz;
	farray phigrax_air, phigray_air, phigraz_air;
	char renderpartiletype;
	float surfacetensionsigma;
	uint *preservemarkscan, *preservemark;
	farray fixedHeat;
	float *parsolubility, *pargascontain, *tempsolubility, *tempgascontain;	//存储粒子中可以容纳的气体量以及现在含有的气体量
	float initgasrate, Temperature0, initdissolvegasrate;
	float vaporGenRate;
	float dragParamSolo, dragParamGrid;
	float viscosiySPH;
	float *pardens, *parpress;
	//heat parameters.
	float meltingpoint, boilingpoint;
	float defaulttemperature;
	float heatalphafluid, heatalphaair, heatalphasolid, heatalphavacuum;
	float *HeatAlphaArray;
	float defaultSolidT, defaultLiquidT, LiquidHeatTh;
	float alphaTempTrans;
	float bounceVelParam, bouncePosParam;
	float heatIncreaseBottom;
	float maxVelForBubble;

	//for render
	float temperatureMax_render, temperatureMin_render;

	//使用预计算好的位置根据温度和溶解度生成empty气泡，当气泡大于一定体积时，生成AIR粒子。
	//对其它模块的影响：markgrid, correctpos, heattransfer.
	void initEmptyBubbles();
	float3 *pEmptyPos, *pEmptyDir;
	float *pEmptyRadius;
	int pEmptyNum;
	//初始化seed的网格。注意是以网格为单位，而不是具体的位置
	void initSeedCell();
	void updateSeedCell();
	int *dseedcell, seednum;

	bool boutputpovray, bOutputColoredParticle,boutputobj;
	int outputframeDelta;		//控制每隔x帧输出一次
	void outputPovRaywater(int frame, float3* dpos, float3 *dnormal, int pnum, uint *dindices, int indicesnum, const char* objectname);
	void outputSoloBubblePovRay(int frame, float3 *dpos, float *dmass, char *dflag, int pnum);
	void outputAirParticlePovRay(int frame, float3 *dpos, float *dmass, char *dflag, int pnum);
	void outputEmptyBubblePovRay(int frame);
	void outputColoredParticle(int frame, float3* dpos, float *ptemperature, int pnum);		//将粒子以sphere的格式输出，位置、颜色和半径。颜色使用mapColorBlue2Red函数按温度进行编码。
	
	void outputOBJwater(int frame, float3* dpos, float3 *dnormal, int pnum, uint *dindices, int indicesnum, const char* objectname);
	void outputSoloBubbleOBJ(int frame, float3 *dpos, float *dmass, char *dflag, int pnum);
	void outputAirParticleOBJ(int frame, float3 *dpos, float *dmass, char *dflag, int pnum);
	void outputEmptyBubbleOBJ(int frame);
	void outputBOBJwater(int frame, float3* dpos, float3 *dnormal, int pnum, uint *dindices, int indicesnum, const char* objectname);//bobj.gz for blender

	//流固耦合的接口，以后尽量加到这一块
	void waterSolidSim();
	void solidmotion();		///////////////////////////////////////////////	计算刚体运动6692963GY
	void solidmotion_fixed();
	void initSolidBuffers();//////////////////////////////////////////////////////////////////GY
	void rendersphereshader();
	void readdata_solid();		//In this program the sphere means the solid model such as sphere!
	void initparticle_solidCoupling();
	void mat4_set_rotate(matrix4* rot, matrix4* m, float angle, float x, float y, float z);
	void mat4_mul(matrix4* dst, const matrix4* m0, const matrix4* m1);
	void mat4_mulvec3_as_mat3(float3* dst, const matrix4* m, const float3* v);
	float3 accumulate_GPU_f3(float3 *data);
	float3 accumulate_CPU_f3_test(float3 *data);
	void CollisionSolid();
	void genAirFromSolid();

	float3 m_bubblepos;
	float m_bubbleradius;

	//for solid particle
	//	float3 *initialSolPos;
	bool m_bSolid;	//标记是否有solid，在Load以及仿真时控制是否调用其对应的功能函数
	bool m_bMelt;	//是否要熔化固体
	bool m_bFreeze;
	bool m_bFixSolid;	//标记是否要把固体固定不动，即不再更新其位置与速度。
	bool m_bGenGas;	//是否产生气泡
	bool m_bHeatTrans;	//是否模拟温度的传递
	bool m_bAddHeatBottom;		//是否在底部增加热
	bool m_bExtendHeatToBoundary;	//向边界一层扩展热场
	bool m_bCorrectPosition;
	bool m_bCPURun;	//控制使用CPU仿真一部分功能。
	int updateSeedFrameDelta;
	bool m_bCorrectFluidByAirLS;
	float buoyanceRateAir, buoyanceRateSolo;
	int m_beginFrame;
	bool m_bBottomParticel;

	int nInitSolPoint, Pointnum, nRealSolpoint;
	double **SolpointPos;		//cpu数据
	float3 rg0;
	float buoyantHeight;			//水面高度，算浮力
	float3* c;
	float3* I;
	float3* solidParPos;
	float3* solidParVelFLIP;
	farray phisolid, solidux, soliduy, soliduz;		//to deal with collision.
	float SolidbounceParam, SolidBuoyanceParam;		//固体碰到边界时反弹的参数，浮力参数
	int NXMC, NYMC, NZMC;
	int maxsolidvert, maxsolidtri;
	float bubbleMaxVel;

	float3 centertmp;
	bool mRecordImage;

	//for cpu computation 
	void compTpChange_CPU();
	void advect_bubble_CPU();
	float3 getParticleVelFromGrid(float3 pos, farray ux, farray uy, farray uz);
	void mapvelp2g_bubble_CPU();
	farray hwaterux, hwateruy, hwateruz;
	float3 *hpos, *hvel;
	float *hmass;
	char *hparflag;
	float *hparLHeat;
	uint *hgridstart, *hgridend;

};



#endif
