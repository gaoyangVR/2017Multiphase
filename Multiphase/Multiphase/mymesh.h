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

	void Load(const char* filename);		//�ο�loadSurfaceOBJ��myLoadObjMesh
	void Draw();
	void AllocateForHash();

	void initGL();
	float3 mynormalize(float3 nor);

	void LoadWithNor(const char* filename); ///////////

public:
	float3* m_hPoints;		//�����
	float3* m_dPoints;		//�����
	uint3* m_hFaces;		//�����
	uint3* m_dFaces;		//�����
	uint3* m_dFacesSorted;
	float3* m_hNormals;		//����䣬���㷨��
	float3* m_dNormals;		//����䣬���㷨��
	float3* m_dHashPointsForFaces;
	int m_nPoints;		//�����
	int m_nFaces;		//�����

	float3 m_max;		//�洢ģ�Ͱ�Χ�е����ֵ��ע�����߸�Ԥ��һ�㣬�����
	float3 m_min;		//�洢ģ�Ͱ�Χ�е���Сֵ��ע�����߸�Ԥ��һ�㣬�����

	//for hashing
	uint* m_dTriHash_radix[2];		//point grid hash, ���飬�����Ԫ����ָ��
	uint* m_dTriCellStart;
	uint* m_dTriCellEnd;
	float3* m_dFaceNormals;		//��������ݣ�������ķ��ߣ���Ҫ������reorderTriangle_radix_q��֮����£�����m_dFacesSorted������

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
