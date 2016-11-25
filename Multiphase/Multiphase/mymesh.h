#ifndef MYMESH_H
#define MYMESH_H
#include <vector_types.h>

typedef unsigned int uint;

class myMesh
{
public:
	myMesh()
	{
		m_nPoints = 0;
		m_nFaces = 0;
		m_bTexed = false;
	}

	void Load(const char* filename);		//参考loadSurfaceOBJ和myLoadObjMesh
	void Draw();
	void AllocateForHash();

	void initGL();
	float3 mynormalize(float3 nor);

	void LoadWithNor(const char* filename); ///////////

public:
	float3* m_hPoints;		//待填充
	float3* m_dPoints;		//待填充
	uint3* m_hFaces;		//待填充
	uint3* m_dFaces;		//待填充
	uint3* m_dFacesSorted;
	float3* m_hNormals;		//待填充，顶点法线
	float3* m_dNormals;		//待填充，顶点法线
	float3* m_dHashPointsForFaces;
	int m_nPoints;		//待填充
	int m_nFaces;		//待填充

	float3 m_max;		//存储模型包围盒的最大值，注意两边各预留一点，待填充
	float3 m_min;		//存储模型包围盒的最小值，注意两边各预留一点，待填充

	//for hashing
	uint* m_dTriHash_radix[2];		//point grid hash, 数组，数组的元素是指针
	uint* m_dTriCellStart;
	uint* m_dTriCellEnd;
	float3* m_dFaceNormals;		//待填充内容：各个面的法线，需要在排序（reorderTriangle_radix_q）之后更新，根据m_dFacesSorted来计算

	unsigned int m_Vao;
	unsigned int m_VertVbo;
	unsigned int m_NormVbo;
	unsigned int m_ElemVbo;
	unsigned int m_TexCoordVbo;
	float2* m_hTexCoords;
	unsigned int m_ColorTex;
	unsigned int m_DepthTex;

	unsigned int m_Fbo;

	bool m_bTexed;
};

#endif
