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
