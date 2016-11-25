#include "mymesh.h"
#include <cuda_runtime_api.h>
#include <helper_math.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>

void myMesh::AllocateForHash()
{
	cudaMalloc((void**)&m_dFacesSorted, sizeof(uint3)*m_nFaces);
	cudaMalloc((void**)&m_dHashPointsForFaces, sizeof(float3)*m_nFaces);
	cudaMalloc((void**)&m_dTriHash_radix[0], sizeof(uint2)*m_nFaces);
	cudaMemset(m_dTriHash_radix[0], 0, sizeof(uint2)*m_nFaces);
	cudaMalloc((void**)&m_dTriHash_radix[1], sizeof(uint2)*m_nFaces);
	cudaMalloc((void**)&m_dFaceNormals, sizeof(float3)*m_nFaces);
}

float3 myMesh::mynormalize(float3 nor)
{
	float temp;
	float3 fnor = nor;
	temp = sqrt(nor.x*nor.x + nor.y*nor.y + nor.z*nor.z);
	if (temp > 0){
		fnor.x = nor.x / temp;
		fnor.y = nor.y / temp;
		fnor.z = nor.z / temp;
	}
	return fnor;
}

void myMesh::initGL()
{
	glGenVertexArrays(1, &m_Vao);

	glGenBuffers(1, &m_VertVbo);
	glGenBuffers(1, &m_NormVbo);
	glGenBuffers(1, &m_ElemVbo);
	if (m_bTexed)
		glGenBuffers(1, &m_TexCoordVbo);

	glBindVertexArray(m_Vao);

	glBindBuffer(GL_ARRAY_BUFFER, m_VertVbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*m_nPoints, m_hPoints, GL_DYNAMIC_DRAW);
	// 	glEnableVertexAttribArray(0);
	// 	glVertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE,0,0);	

	glBindBuffer(GL_ARRAY_BUFFER, m_NormVbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*m_nPoints, m_hNormals, GL_DYNAMIC_DRAW);
	// 	glEnableVertexAttribArray(1);
	// 	glVertexAttribPointer (1, 3, GL_FLOAT, GL_FALSE,0,0);

	if (m_bTexed)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_TexCoordVbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float2)*m_nPoints, m_hTexCoords, GL_DYNAMIC_DRAW);
		// 		glEnableVertexAttribArray(2);
		// 		glVertexAttribPointer (2, 2, GL_FLOAT, GL_FALSE,0,0);
	}

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ElemVbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint3)*m_nFaces, m_hFaces, GL_STATIC_DRAW);

	glBindVertexArray(0);

	glGenFramebuffers(1, &m_Fbo);

	glBindFramebuffer(GL_FRAMEBUFFER, m_Fbo);

	glReadBuffer(GL_NONE);
	GLenum draws[1];
	draws[0] = GL_COLOR_ATTACHMENT0;
	glDrawBuffers(1, draws);
	glBindTexture(GL_TEXTURE_2D, m_DepthTex);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_DepthTex, 0);
	glBindTexture(GL_TEXTURE_2D, m_ColorTex);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_ColorTex, 0);

	GLenum FBOstatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (FBOstatus != GL_FRAMEBUFFER_COMPLETE) {
		printf("GL_FRAMEBUFFER_COMPLETE failed, CANNOT use FBO\n");
	}

	//XIAQING:转回窗口系统提供的framebuffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void myMesh::Draw()
{
	if (m_nPoints == 0 || m_nFaces == 0)
		return;

	//glBindVertexArray(m_Vao);

	glUseProgram(0);

	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	// 	glEnable(GL_AUTO_NORMAL); 
	// 	glEnable(GL_NORMALIZE); 
	glDisable(GL_BLEND);
	//	glDisable( GL_COLOR_MATERIAL );
	glBindBuffer(GL_ARRAY_BUFFER, m_VertVbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, m_NormVbo);
	glNormalPointer(GL_FLOAT, 0, 0);
	glEnableClientState(GL_NORMAL_ARRAY);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ElemVbo);

	glDrawElements(GL_TRIANGLES, m_nFaces * 3, GL_UNSIGNED_INT, 0);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);

	//	glEnable( GL_COLOR_MATERIAL );
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}
void myMesh::LoadWithNor(const char* filename)////////////////////////////////
{
	FILE *fp;

	if ((fp = fopen(filename, "r")) != NULL)
		printf("load obj successfully!\n");

	char str[100];

	int vertCount = 0;
	int faceCount = 0;
	int normCount = 0;

	while (fgets(str, 100, fp) != NULL)
	{
		if (str[0] == 'v')
		{
			if (str[1] == ' ')
				vertCount++;
			else if (str[1] == 'n')
				normCount++;
		}
		else if (str[0] == 'f')
		{
			faceCount++;
		}
	}

	float3* tmpPoints = (float3*)malloc(sizeof(float3)*vertCount);
	uint3* tmpFaces = (uint3*)malloc(sizeof(uint3)*faceCount);
	float3* tmpNormals = (float3*)malloc(sizeof(float3)*normCount);
	uint3* tmpNorIndex = (uint3*)malloc(sizeof(uint3)*faceCount);

	m_nFaces = faceCount;
	m_nPoints = m_nFaces * 3;
	
	m_hPoints = (float3*)malloc(sizeof(float3)*m_nPoints);
	m_hFaces = (uint3*)malloc(sizeof(uint3)*m_nFaces);
	m_hNormals = (float3*)malloc(sizeof(float3)*m_nPoints);

	rewind(fp);

	int pointsIndex = 0;
	int facesIndex = 0;
	int normalsIndex = 0;

	m_min = make_float3(100.0);
	m_max = make_float3(-100.0);;

	while (fgets(str, 100, fp) != NULL)
	{
		if (str[0] == 'v')
		{
			if (str[1] == ' ')
			{
				float tmp1, tmp2, tmp3;
				sscanf(&str[2], " %f %f %f", &tmp1, &tmp2, &tmp3);

				m_min.x = (tmp1<m_min.x) ? tmp1 : m_min.x;
				m_min.y = (tmp2<m_min.y) ? tmp2 : m_min.y;
				m_min.z = (tmp3<m_min.z) ? tmp3 : m_min.z;

				m_max.x = (tmp1>m_max.x) ? tmp1 : m_max.x;
				m_max.y = (tmp2>m_max.y) ? tmp2 : m_max.y;
				m_max.z = (tmp3>m_max.z) ? tmp3 : m_max.z;

				tmpPoints[pointsIndex].x = tmp1;
				tmpPoints[pointsIndex].y = tmp2;
				tmpPoints[pointsIndex].z = tmp3;
				pointsIndex++;
			}
			else if (str[1] == 'n')
			{
				float tmp1, tmp2, tmp3;
				sscanf(&str[2], " %f %f %f", &tmp1, &tmp2, &tmp3);

				tmpNormals[normalsIndex].x = tmp1;
				tmpNormals[normalsIndex].y = tmp2;
				tmpNormals[normalsIndex].z = tmp3;
				normalsIndex++;
			}
		}
		else if (str[0] == 'f')
		{
			int tmp1, tmp2, tmp3;
			int tp1, tp2, tp3;
			sscanf(&str[1], " %d\/\/%d %d\/\/%d %d\/\/%d", &tmp1, &tp1, &tmp2, &tp2, &tmp3, &tp3);
			tmpFaces[facesIndex].x = tmp1 - 1;
			tmpFaces[facesIndex].y = tmp2 - 1;
			tmpFaces[facesIndex].z = tmp3 - 1;


			tmpNorIndex[facesIndex].x = tp1 - 1;
			tmpNorIndex[facesIndex].y = tp2 - 1;
			tmpNorIndex[facesIndex].z = tp3 - 1;

			facesIndex++;
		}
	}

	//scale
	float fscale = 1.0f / (m_max.x - m_min.x);
	float3 offset = make_float3(0.5f, 0.5f, -m_min.z*fscale + 1.0f / 64);

	for (int i = 0; i < faceCount; i++)
	{
		uint3 face = tmpFaces[i];

		float3 v1 = tmpPoints[face.x] * fscale + offset;
		float3 v2 = tmpPoints[face.y] * fscale + offset;
		float3 v3 = tmpPoints[face.z] * fscale + offset;

		m_hPoints[3 * i] = v1;
		m_hPoints[3 * i + 1] = v2;
		m_hPoints[3 * i + 2] = v3;

		m_hFaces[i].x = 3 * i;
		m_hFaces[i].y = 3 * i + 1;
		m_hFaces[i].z = 3 * i + 2;

		uint3 norm = tmpNorIndex[i];

		float3 n1 = tmpNormals[norm.x];
		float3 n2 = tmpNormals[norm.y];
		float3 n3 = tmpNormals[norm.z];

		m_hNormals[3 * i] = n1;
		m_hNormals[3 * i + 1] = n2;
		m_hNormals[3 * i + 2] = n3;

	}

	cudaMalloc((void**)&m_dPoints, sizeof(float3)*m_nPoints);
	cudaMalloc((void**)&m_dFaces, sizeof(float3)*m_nFaces);
	cudaMalloc((void**)&m_dNormals, sizeof(uint3)*m_nPoints);

	cudaMemcpy(m_dPoints, m_hPoints, sizeof(float3)*m_nPoints, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dFaces, m_hFaces, sizeof(float3)*m_nFaces, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dNormals, m_hNormals, sizeof(uint3)*m_nPoints, cudaMemcpyHostToDevice);
}