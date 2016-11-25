#include "utility.h"
#include "device_launch_parameters.h"
void copyparamtoGPU(FlipConstant hparam);
void copyNXNYNZtoGPU(int nx, int ny, int nz);
void copyNXNYNZtoGPU_MC(int nx, int ny, int nz);

__global__ void cptdivergence(farray outdiv, farray ux, farray uy, farray uz, charray mark);
__global__ void advectparticle(float3 *pos, float3 *vel, int pnum, farray ux, farray uy, farray uz, float dt, char *parflag, VELOCITYMODEL velmode);
__global__ void advectparticle_RK2(float3 *ppos, float3 *pvel, int pnum, farray ux, farray uy, farray uz, float dt,
	char *parflag, VELOCITYMODEL velmode);
__global__ void advectparticle_waterfall_RK2(float3 *ppos, float3 *pvel, int pnum, farray ux, farray uy, farray uz, float dt,
	char *parflag, VELOCITYMODEL velmode, float solidx, float solidz);
__global__ void mapvelp2g_slow(float3 *pos, float3 *vel, int pnum, farray ux, farray uy, farray uz);
__global__ void mapvelp2g_hash(float3 *pos, float3 *vel, float *mass, char *parflag, int pnum, farray ux, farray uy, farray uz, uint* gridstart, uint *gridend);
__global__ void mapvelg2p_flip(float3 *pos, float3 *vel, char* parflag, int pnum, farray ux, farray uy, farray uz);
__global__ void subGradPress(farray p, farray ux, farray uy, farray uz);
__global__ void JacobiIter(farray outp, farray p, farray b, charray mark);
__global__ void markair(charray mark);
__global__ void flipAirVacuum(charray mark);
__global__ void markfluid(charray mark, float3 *pos, char *parflag, int pnum);
__global__ void markfluid_dense(charray mark, float *parmass, char *parflag, int pnum, uint *gridstart, uint *gridend, int fluidParCntPerGridThres);
__global__ void markBoundaryCell(charray mark);
__global__ void setgridcolor_k(float* color, ECOLORMODE mode, farray p, farray ux, farray uy, farray uz, farray div, farray phi, charray mark, farray ls, farray tp, float sigma, float temperatureMax, float temperatureMin);
__global__ void initphi(farray phi, charray mark, char typeflag);
__global__ void sweepphi(farray phi);
__global__ void sweepphibytype(farray phi, charray mark, char typeflag);
__global__ void sweepu(farray outux, farray outuy, farray outuz, farray ux, farray uy, farray uz, farray phi, charray mark);
__global__ void setWaterBoundaryU_k(farray ux, farray uy, farray uz, charray mark);
__global__ void setSmokeBoundaryU_k(farray ux, farray uy, farray uz, charray mark);
__global__ void computeDeltaU(farray ux, farray uy, farray uz, farray uxold, farray uyold, farray uzold);
__global__ void addgravityforce_k(float3 *vel, char* parflag, int pnum, float dt);
__global__ void setPressBoundary(farray press);
__global__ void correctparticlepos(float3* outpos, float3* ppos, float *pmass, char* parflag, int pnum, uint* gridstart, uint *gridend, float correctionspring, float correctionradius, float3 *pepos, float *peradius, int penum);
__global__ void copyParticle2GL_radius_k(float3* ppos, float *pmass, char *pflag, int pnum, float *renderpos, float *rendercolor, float minmass);
__global__ void copyParticle2GL_vel_k(float3* ppos, float3 *pvel, float *pmass, char *pflag, int pnum, float *renderpos, float *rendercolor);
__global__ void cptdivergence_bubble2(farray outdiv, farray waterux, farray wateruy, farray wateruz, farray airux, farray airuy, farray airuz, charray mark, farray ls);


//for smoke
__global__ void advectux(farray outux, farray ux, farray uy, farray uz, float velocitydissipation, float3 wind);
__global__ void advectuy(farray outuy, farray ux, farray uy, farray uz, float velocitydissipation, float3 wind);
__global__ void advectuz(farray outuz, farray ux, farray uy, farray uz, float velocitydissipation, float3 wind);
__global__ void advectscaler(farray outscalar, farray scalar, farray ux, farray uy, farray uz, float densedissipation, float3 wind);
__global__ void markforsmoke(charray mark, farray spraydense);
__global__ void setsmokevel(farray uz, farray dense);
__global__ void setsmokedense(farray dense);
__global__ void setsmokevel_nozzle(farray ux, farray dense);
void writedens2surface(cudaArray* cudaarray, int blocknum, int threadnum, farray dense);
__global__ void diffuse_dense(farray outp, farray inp, charray mark, float alpha, float beta);
__global__ void diffuse_velocity(farray outv, farray inv, float alpha, float beta);

//pcg: preconditioned conjugate gradient method
__global__ void arrayproduct_k(float* out, float* x, float *y, int n);
__global__ void computeAx(farray ans, charray mark, farray x, int n);
__global__ void pcg_op(charray A, farray ans, farray x, farray y, float a, int n);
__global__ void buildprecondition_pcg(farray P, charray mark, farray ans, farray input, int n);


//for sort the particles.
__global__ void calcHashD(uint*   gridParticleHash, uint*   gridParticleIndex, float3* pos, uint numParticles);
__global__ void calcHashD_MC(uint*   gridParticleHash, uint*   gridParticleIndex, float3* pos, uint numParticles);
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
	uint    numParticles);

//Marching Cube.
void allocateTextures(uint **d_edgeTable, uint **d_triTable, uint **d_numVertsTable);
// compact voxel array
__global__ void genWaterDensfield(farray outdens, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, float fMCDensity);
__global__ void genWaterDensfield2(farray outdens, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, float fMCDensity, char MCParType);
__global__ void genWaterDensfield_liquidAndGas(farray outdens, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, float fMCDensity);
__global__ void genWaterDensfield_Gas(farray outdens, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, float fMCDensity, SCENE scene);
__global__ void genWaterDensfield_GY(farray outdens, float3 *pos, char *parflag, uint *gridstart, uint  *gridend, float fMCDensity, char MCParType, float3 centertmp);
__global__ void classifyVoxel(uint* voxelVerts, uint *voxelOccupied, farray volume, float isoValue);
__global__ void compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels);
__global__ void generateTriangles2(float3 *pos, float3 *norm, uint *compactedVoxelArray, uint *numVertsScanned, farray volume,
	float isoValue, uint activeVoxels, uint maxVerts);
__global__ void markActiveEdge_MC(uint *outmark, uint *compactedVoxelArray, farray volume, float isoValue, uint activeVoxels);
__global__ void generateTriangles_indices(float3 *pTriVertex, uint *pTriIndices, uint *compactedVoxelArray, farray volume,
	float isoValue, uint activeVoxels, uint maxVerts, uint *MCEdgeIdxMapped, uint *vertexnumscan);
__global__ void calnormal_k(float3 *ppos, float3 *pnor, int pnum, uint *indices, int indicesnum);
__global__ void normalizeTriangleNor_k(float3 *pnor, int pnum);
__global__ void genSphereDensfield(farray outdens, float3 center, float radius);
//smoothing 3d triangle mesh
__global__ void smooth_computedisplacement(float3 *displacement, int *weight, float3 *ppos, uint *indices, int trianglenum);
__global__ void smooth_addDisplacement(float3 *displacement, int *weight, float3 *ppos, int vertexnum, float param);
//rendering spray particle, map the mass into density field.
__global__  void sprayparticle2density(float3 *ppos, float *pmass, char *pflag, int pnum, farray density, farray indensity, char *pflagKeep, float transDensityThreshold);

//solid coupling
__global__ void markSolid_sphere(float3 spherepos, float sphereradius, charray mark);
__global__ void collidesphere(float3 spherepos, float sphereradius, float3* ppos, float3 *pvel, int pnum, float sphereBounceParam);
__global__ void markSolid_waterfall(int3 minpos, int3 maxpos, charray mark);
__global__ void markSolid_waterfall_liquid(int3 minpos, int3 maxpos, charray mark);
__global__ void markSolid_terrain(charray mark, charray mark_terrain);

__global__ void createAABB_q(float3* points, int nPoints, uint3* faces, int nFaces, float *maxLength, float3* hashPoints);
__global__	void calcHash_radix_q(
	uint2*   gridParticleIndex, // output
	float3* posArray,               // input: positions
	uint    numParticles,
	float3 t_min,
	float3 t_max);
__global__ void reorderDataAndFindCellStart_radix_q(uint*   cellStart,        // output: cell start index
	uint*   cellEnd,          // output: cell end index
	uint3* sortedFaces,
	uint2 *  gridParticleHash, // input: sorted grid hashes
	uint3* oldFaces,
	uint    numParticles);

__global__ void calculateNormal(float3* points, uint3* faces, float3* normals, int num);
__global__ void Coupling_f2s_k_q(float3 *ppos, float3 *pvel, float *pmass, int numFPnt, float3* surPoints, float3* surfaceNor,
	uint3* surfaceIndex, int surfaceNum, uint *cellStart, uint *cellEnd,
	float3 t_min, float3 t_max, int scene);

//for bubbles.
__global__ void addbuoyancyforce_k(float dheight, float3 *pos, float3 *vel, char* parflag, int pnum, float dt);
__global__ void addbuoyancyforce_vel(float velMax, float3 *pos, float3 *vel, char* parflag, int pnum, float dt, float buoyanceRateAir, float buoyanceRateSolo);
__global__ void genlevelset(farray lsfluid, farray lsair, charray mark, float3 *pos, char *parflag, float *pmass, uint *gridstart, uint  *gridend, float fMCDensity, float offset);
__global__ void mapvelp2g_k_air(float3 *pos, float3 *vel, float *mass, char *parflag, int pnum, farray ux, farray uy, farray uz, uint* gridstart, uint *gridend);
__global__ void mapvelp2g_k_fluidSolid(float3 *pos, float3 *vel, float *mass, char *parflag, int pnum, farray ux, farray uy, farray uz, uint* gridstart, uint *gridend);
__global__ void mapvelp2g_k_solid(float3 *pos, float3 *vel, float *mass, char *parflag, int pnum, farray ux, farray uy, farray uz, uint* gridstart, uint *gridend);
__global__ void cptdivergence_bubble(farray outdiv, farray waterux, farray wateruy, farray wateruz, farray airux, farray airuy, farray airuz, charray mark, farray ls, farray sf);
__global__ void computeAx_bubble(farray ans, charray mark, farray x, int n);
__global__ void pcg_op_bubble(charray A, farray ans, farray x, farray y, float a, int n);
__global__ void advectparticle_RK2_bubble(float3 *ppos, float3 *pvel, int pnum, farray waterux, farray wateruy, farray wateruz,
	farray airux, farray airuy, farray airuz, float dt, char *parflag, VELOCITYMODEL velmode);
__global__ void mapvelg2p_flip_bubble(float3 *ppos, float3 *vel, char* parflag, int pnum, farray waterux, farray wateruy, farray wateruz, farray airux, farray airuy, farray airuz);
//__global__ void compsurfacetension_k( farray sf, farray ls );
__global__ void compsurfacetension_k(farray sf, charray mark, farray phigrax, farray phigray, farray phigraz, float sigma);
__global__ void preparels(farray ls, charray mark);
__global__ void setLSback(farray ls);
__global__ void markLS_bigpositive(farray ls, charray mark);
__global__ void setLSback_bigpositive(farray ls);
__global__ void mergeLSAndMarkGrid(farray lsmerge, charray mark, farray lsfluid, farray lsair);
__global__ void sweepu_k_bubble(farray outux, farray outuy, farray outuz, farray ux, farray uy, farray uz, farray ls, charray mark, char sweepflag);
__global__ void correctbubblepos(farray ls, farray phigrax, farray phigray, farray phigraz, float3 *ppos, char* pflag, int pnum, float *pphi);
__global__ void correctbubblepos_air(farray lsmerge, farray phigrax, farray phigray, farray phigraz, farray lsair, farray phigrax_air, farray phigray_air, farray phigraz_air, float3 *ppos, char* pflag, int pnum, float *pphi);
__global__ void computePhigra(farray phigrax, farray phigray, farray phigraz, farray ls);
__global__ void copyParticle2GL_phi(float3* ppos, char *pflag, float *pmass, float *pTemperature, int pnum, float *renderpos, float *rendercolor, farray ls, farray phigrax, farray phigray, farray phigraz, char typeflag, float Tmax, float Tmin);
__global__ void subGradPress_bubble(farray p, farray ux, farray uy, farray uz, farray sf, farray lsmerge, charray mark);
__global__ void enforcesurfacetension_p(float3* ppos, float3 *pvel, char *pflag, int pnum, farray lsmerge, farray sf, farray phigrax, farray phigray, farray phigraz, charray mark, SCENE scene);
__global__ void sweepVacuum(charray mark);
__global__ void markDeleteAirParticle(float3* ppos, char* pflag, float *pmass, uint *preservemark, int pnum, charray mark, farray lsmerge, farray lsair, uint *cnt);
__global__ void deleteparticles(uint *preserveflag, uint *preserveflagscan, int pnum, float3 *outpos, float3 *pos,
	float3 *outvel, float3 *vel, float *outmass, float* mass, char *outflag, char *flag, float *outTemperature, float *temperature, float *outheat, float *heat,
	float *outsolubility, float *solubility, float *outgascontain, float *gascontain);
__global__ void enforceForceSoloAirP(float3 *ppos, float3 *pvel, float *pdens, float *ppress, char *pflag, int pnum, uint *gridstart, uint *gridend, float viscositySPH, float maxVelForBubble);
__global__ void verifySoloAirParticle(float3 *ppos, float3 *pvel, char *pflag, int pnum, farray lsmerge, farray airux, farray airuy, farray airuz, uint *gridstart, uint *gridend, SCENE scene);
__global__ void updatesolubility(float *psolubility, float *ptemperature, char *pflag, int pnum, float Solubility0, float Temperature0, float dissolvegasrate);
__global__ void GenerateGasParticle_k(float *psolubility, float *paircontain, float3 *ppos, float3 *pvel, float *pmass, char *pflag, float *pTemperature, float *pLHeat,
	int pnum, uint *gridstart, uint *gridend, int *addparnums, float *randfloat, int randcnts, int frame, farray gTemperature, float LiquidHeatTh,
	int *seedcell, int seednum, float vaporGenRate);
__global__ void initsolubility_k(float *psolubility, float* pgascontain, float *ptemperature, char *pflag, int pnum, float Solubility0, float Temperature0, float dissolvegasrate, float initgasrate);
__global__ void updateEmptyBubbles(float3 *pepos, float3 *pedir, float *peradius, int penum, float3 *parpos, float3 *parvel, float *parmass, float* parTemperature,
	char *parflag, float *parsolubility, float *paraircontain, int parnum, int *addparnums, uint *gridstart, uint *gridend, farray gTemperature);
__global__ void updatebubblemass(float *psolubility, float *paircontain, float3 *ppos, float *pmass, char *pflag, int pnum, uint *gridstart, uint *gridend);
__global__ void calDragForce(float3 *ppos, float3 *pvel, char *pflag, int pnum, farray ux, farray uy, farray uz, float dragparamsolo, float dragparamgrid, SCENE scene);
__global__ void calcDensPress_Air(float3* ppos, float *pdens, float *ppress, char* pflag, int pnum, uint *gridstart, uint *gridend);


//for heat
__global__ void computeAx_heat(farray ans, charray mark, farray x, int n, float *heatAlphaArray, farray fixedHeat, SCENE scene);
__global__ void pcg_op_heat(charray A, farray ans, farray x, farray y, float a, int n);
__global__ void compb_heat(farray Tp_old, farray Tp, farray fixedheat, charray mark, float *heatAlphaArray);
__global__ void initHeatParticle(float *pTemperature, float *pHeat, float defaultSolidT, float defaultLiquidT, float LiquidHeatTh, char *pflag, int pnum);
__global__ void mapHeatp2g_hash(float3 *ppos, float *pTemperature, int pnum, farray heat, uint* gridstart, uint *gridend, float defaulttemperature);
__global__ void mapHeatg2p(float3 *ppos, char *parflag, float *pTemperature, int pnum, farray Tchange, farray T, float defaultSolidT, float alphaTempTrans);
__global__ void mapHeatg2p_MeltAndBoil(float3 *ppos, char *parflag, float *pTemperature, int pnum, farray Tchange, farray T, float defaultSolidT, float alphaTempTrans);
__global__ void setBoundaryHeat(farray tp);
__global__ void compTpChange(farray tp, farray tpsave, charray mark);
__global__ void updateFixedHeat(farray fixedHeat, int frame);
__global__ void updateLatentHeat_k(float *parTemperature, float *parLHeat, char *partype, int pnum, float meltingpoint, float boilingpoint, float LiquidHeatTh);
__global__ void MeltingSolidByHeat(float *pTemperature, float *pLHeat, char *pflag, int pnum, float LiquidHeatTh, float meltTemperature, int *numchange);
__global__ void FreezingSolidByHeat(float3* ppos, float *pLHeat, char *pflag, int pnum, int *numchange, uint *gridstart, uint *gridend);
__global__ void pouringwater(float3* pos, float3* vel, float* parmass, char* parflag, float *ptemperature, float *pLHeat, float *pGasContain, int parnum,
	float3 *ppourpos, float3 *ppourvel, char pourflag, int pournum, float *randfloat, int randnum, int frame, float posrandparam, float velrandparam,
	float defaultLiquidT, float LiquidHeatTh);
__global__ void addHeatAtBottom(farray Tp, int frame, float heatIncreaseBottom);
__global__ void initheat_grid_k(farray tp, charray mark);

//for GY: solid coupling
__global__ void accumulate_GPU_k(int num, float3* out, float3* a);
__global__ void compute_cI_k(int pnum, char* parflag, float3 *parPos, float3 *parVel, float3* c, float3* i, float3 rg);
__global__ void computeVelSolid_k(float3* parPos, char* parflag, float3* parVel, int pnum, float3 rg, float3 R, float3 T);
__global__ void setVelZeroSolid_k(float3 *parvel, char *parflag, int pnum);
__global__ void set_nonsolid_2_zero(char* parflag, int pnum, float3* Pos, float3* Vel);
__global__ void computePosSolid_k(float3* parPos, char* parflag, int pnum, float3 rg, float3 rg0, matrix3x3 rm);
__global__ void computeSolidVertex_k(float3* vertexpos, int vnum, float3 rg, float3 rg0, matrix3x3 rm);
__global__ void CollisionWithSolid_k(float3 *ppos, float3 *pvel, char *pflag, int pnum, farray phisolid, farray sux, farray suy, farray suz, SCENE scene, float bounceVelParam, float bouncePosParam);
__global__ void CollisionWithSolid_Freezing(float3 *ppos, float3 *pvel, char *pflag, int pnum, farray phisolid, uint* gridstart, uint* gridend);
__global__ void initSolidPhi(farray phi, uint *gridstart, uint *gridend, char *pflag);
__global__ void solidCollisionWithBound(float3 *ppos, float3 *pvel, char *pflag, int pnum, float SolidbounceParam, int nSolPoint);
__global__ void buoyancyForSolid(float3 *ppos, float3 *pvel, char *pflag, int pnum, uint *gridstart, uint *gridend, float SolidBuoyanceParam);
// __global__ void genAirFromSolid_k( float3 *ppos, float3 *pvel, char *pflag, float *psolubility, float *paircontain, float *pmass, float *pTemperature,int pnum, 
//								charray lsmark, farray phisolid, farray Tgrid, int *addnum, float *randfloat, int nrandnum, int frame );

//liquid-solid coupling in SPH framework
__global__ void calcDensPressSPH_SLCouple(float3* ppos, float *pdens, float *ppress, char* pflag, int pnum, uint *gridstart, uint *gridend);
__global__ void enforceForceSPH_SLCouple(float3 *ppos, float3 *pvel, float *pdens, float *ppress, char *pflag, int pnum, uint *gridstart, uint *gridend, float viscositySPH);
