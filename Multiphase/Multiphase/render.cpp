//屏蔽掉unicode的warning.
#pragma warning ( disable : 4819 )

#include<Windows.h>
#include "inc/glm-0.9.4.0/glm/glm.hpp"
#include "inc/glm-0.9.4.0/glm/gtc/matrix_transform.hpp"
#include "spray.h"
#include "GL/glew.h"
#include "GL/gl.h"
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include "loadshader.h"
#include "camera.h"
#include <helper_math.h>
#include "cubevertex.h"
using namespace glm;

extern Camera gcamera;


void setLookAtFromCam()
{
	gluLookAt(gcamera.cam_from.x, gcamera.cam_from.y, gcamera.cam_from.z,
		gcamera.cam_to.x, gcamera.cam_to.y, gcamera.cam_to.z,
		gcamera.cam_up.x, gcamera.cam_up.y, gcamera.cam_up.z);
}

void setProjectionFromCam()
{
	gluPerspective(gcamera.cam_fov, gcamera.cam_aspect, gcamera.nearplane, gcamera.farplane);
}

void setglmLookAtMatrixFromCam(mat4 &mat)
{
	mat = glm::lookAt(vec3(gcamera.cam_from.x, gcamera.cam_from.y, gcamera.cam_from.z),
		vec3(gcamera.cam_to.x, gcamera.cam_to.y, gcamera.cam_to.z),
		vec3(gcamera.cam_up.x, gcamera.cam_up.y, gcamera.cam_up.z));
}

void setglmProjMatrixFromCam(mat4 &mat)
{
	mat = glm::perspective(gcamera.cam_fov, gcamera.cam_aspect, gcamera.nearplane, gcamera.farplane);
}

void cspray::render()
{
	//glClearColor( 1.0f, 1.0f, 1.0f, 1.0f );
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//  	glEnable (GL_LINE_SMOOTH );
	// 	glHint (GL_LINE_SMOOTH, GL_NICEST);
	glLineWidth(2);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	setProjectionFromCam();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	setLookAtFromCam();

	glUseProgram(0);
	glDisable(GL_BLEND);
	//glDisable( GL_LIGHTING );
	glColor3f(1.0f, 0.f, 0.f);

	//render the simple cube boundary
	glEnable(GL_DEPTH_TEST);
	glPushMatrix();
	glTranslatef(hparam.gmax.x*0.5f, hparam.gmax.y*0.5f, hparam.gmax.z*0.5f);
	glScalef(hparam.gmax.x, hparam.gmax.y, hparam.gmax.z);
	glDisable(GL_LIGHTING);
	if (mscene != SCENE_HEATTRANSFER)
		glutWireCube(1.0f);
	glPopMatrix();

	if (bCouplingSphere)
		rendersphere();

	if (simmode == SIMULATION_WATER || simmode == SIMULATION_SOLIDCOUPLING || simmode == SIMULATION_BUBBLE || simmode == SIMULATION_HEATONLY)
	{
		glEnable(GL_PROGRAM_POINT_SIZE);
		if (rendermode == RENDER_PARTICLE || rendermode == RENDER_ALL)
			renderparticleshader();
		else if (rendermode == RENDER_MC)
			renderIsosurface_smooth();
		else if (rendermode == RENDER_GRID || rendermode == RENDER_ALL)
		{
			setGridColor();
			rendergridvariables();
		}
	}
	else if (simmode == SIMULATION_SMOKE)
	{
		if (rendermode == RENDER_PARTICLE || rendermode == RENDER_ALL)
			smokeRayCasting();
		if (rendermode == RENDER_GRID || rendermode == RENDER_ALL)
		{
			setGridColor();
			rendergridvariables();
		}
	}

	glutPostRedisplay();
	glutSwapBuffers();
}

void cspray::initlight()
{
	// good old-fashioned fixed function lighting
	float black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	float ambient[] = { 0.1f, 0.1f, 0.1f, 1.0f };
	float diffuse[] = { 0.9f, 0.9f, 0.9f, 1.0f };
	float lightPos[] = { 0.0f, 0.0f, 1.0f, 0.0f };

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);

	glLightfv(GL_LIGHT0, GL_AMBIENT, white);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
	glLightfv(GL_LIGHT0, GL_SPECULAR, white);
	glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);

	glEnable(GL_LIGHT0);
}

void cspray::initcubeGLBuffers()
{
	glGenVertexArrays(1, &vaocube);
	glBindVertexArray(vaocube);
	glGenBuffers(1, &vbocube);
	glBindBuffer(GL_ARRAY_BUFFER, vbocube);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)* 36 * 3, g_vertex_buffer_data, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void cspray::smokeRayCasting()
{
	glUseProgram(mProgramSmoke);
	glBindVertexArray(vaocube);

	//set uniform variables.
	glm::mat4 model = glm::mat4(1.0f), view, proj;
	setglmLookAtMatrixFromCam(view);
	setglmProjMatrixFromCam(proj);
	mat4 MVP = proj*view*model;		//这里的顺序很重要
	glUniformMatrix4fv(glGetUniformLocation(mProgramSmoke, "MVP"), 1, GL_FALSE, &MVP[0][0]);
	glUniform3f(glGetUniformLocation(mProgramSmoke, "eyepos"), gcamera.cam_from.x, gcamera.cam_from.y, gcamera.cam_from.z);
	//printf("eyepos=%f,%f,%f\n", gcamera.cam_from.x, gcamera.cam_from.y, gcamera.cam_from.z );
	glUniform3f(glGetUniformLocation(mProgramSmoke, "lightpos"), 2.0f, 0.5f, 0.5f);
	//glPolygonMode(GL_FRONT, GL_LINE) ;
	// 	glPolygonMode(GL_BACK, GL_LINE) ;
	//glFrontFace(GL_CCW);
	glEnable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glBindTexture(GL_TEXTURE_3D, densTex3DID);

	glDrawArrays(GL_TRIANGLES, 0, 12 * 3);

	glBindVertexArray(0);
	glUseProgram(0);
}

void cspray::loadshader()
{
	mProgramParticle = glCreateProgram();
	Shader::attachAndLinkProgram(mProgramParticle, Shader::loadShaders("particle.vert", "particle.frag"));
	mProgramGrid = glCreateProgram();
	Shader::attachAndLinkProgram(mProgramGrid, Shader::loadShaders("grid.vert", "grid.frag"));
//	mProgramSmoke = glCreateProgram();
//	Shader::attachAndLinkProgram(mProgramSmoke, Shader::loadShaders("smoke.vert", "smoke.frag"));
}

void cspray::renderparticleshader()
{
	glUseProgram(mProgramParticle);

	glBindVertexArray(vaoPar);

	//set uniform variables.
	glm::mat4 model = glm::mat4(1.0f), view, proj;
	setglmLookAtMatrixFromCam(view);
	setglmProjMatrixFromCam(proj);
	mat4 MVP = proj*view*model;		//这里的顺序很重要
	mat4 MV = view*model;
	glUniform1f(glGetUniformLocation(mProgramParticle, "pointScale"), 600.0f / tanf(35.0f*0.5f*(float)M_PI / 180.0f));
	glUniformMatrix4fv(glGetUniformLocation(mProgramParticle, "MVP"), 1, GL_FALSE, &MVP[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(mProgramParticle, "MV"), 1, GL_FALSE, &MV[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(mProgramParticle, "u_Persp"), 1, GL_FALSE, &proj[0][0]);

	glEnable(GL_DEPTH_TEST);
	glDrawArrays(GL_POINTS, 0, parNumNow);

	glBindVertexArray(0);
	glUseProgram(0);
}

void cspray::initParticleGLBuffers()
{
	glGenVertexArrays(1, &vaoPar);
	glBindVertexArray(vaoPar);

	glGenBuffers(1, &vboParPos);
	glBindBuffer(GL_ARRAY_BUFFER, vboParPos);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)* 3 * parNumMax, 0, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	cudaGLRegisterBufferObject(vboParPos);

	glGenBuffers(1, &vboParColor);
	glBindBuffer(GL_ARRAY_BUFFER, vboParColor);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)* 3 * parNumMax, 0, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	cudaGLRegisterBufferObject(vboParColor);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void cspray::rendergridvariables()
{
	glUseProgram(mProgramGrid);
	glBindVertexArray(vaoGrid);

	//set uniform variables.
	glm::mat4 model = glm::mat4(1.0f), view, proj;
	setglmLookAtMatrixFromCam(view);
	setglmProjMatrixFromCam(proj);
	mat4 MVP = proj*view*model;		//这里的顺序很重要
	mat4 MV = view*model;
	glUniformMatrix4fv(glGetUniformLocation(mProgramGrid, "MVP"), 1, GL_FALSE, &MVP[0][0]);

	glDrawArrays(GL_POINTS, 0, hparam.gnum);

	glBindVertexArray(0);
	glUseProgram(0);
}

void cspray::initGridGLBuffers()
{
	//initialize the grid position, never change.
	int n = hparam.gnum;
	float *pos = new float[n * 3];
	float *color = new float[n * 3];
	int i, j, k;
	for (int idx = 0; idx < n; idx++)
	{
		getijk(i, j, k, idx, NX, NY, NZ);
		pos[idx * 3] = (i + 0.5f)*hparam.cellsize.x;
		pos[idx * 3 + 1] = (j + 0.5f)*hparam.cellsize.x;
		pos[idx * 3 + 2] = (k + 0.5f)*hparam.cellsize.x;
		color[idx * 3] = color[idx * 3 + 1] = color[idx * 3 + 2] = 0.5f;
	}

	//init opengl buffers.
	glGenVertexArrays(1, &vaoGrid);
	glBindVertexArray(vaoGrid);

	glGenBuffers(1, &vboGridpos);
	glBindBuffer(GL_ARRAY_BUFFER, vboGridpos);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)* 3 * n, pos, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glGenBuffers(1, &vboGridcolor);
	glBindBuffer(GL_ARRAY_BUFFER, vboGridcolor);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*n * 3, color, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	cudaGLRegisterBufferObject(vboGridcolor);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	delete[] pos;
	delete[] color;
}

void cspray::initdensityGLBuffers()
{
	glGenTextures(1, &densTex3DID);
	glBindTexture(GL_TEXTURE_3D, densTex3DID);
	{
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, NX, NY, NZ, 0, GL_RGBA, GL_FLOAT, NULL);
	}
	glBindTexture(GL_TEXTURE_3D, 0);
	cudaGraphicsGLRegisterImage(&densTex3D_cuda, densTex3DID, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void
createVBO(GLuint* vbo, unsigned int size)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glutReportErrors();
}

void cspray::initMCtriangles(int _maxVerts, int _maxTriangles)
{
	// create VBOs
	createVBO(&posVbo, _maxVerts*sizeof(float)* 3);
	// DEPRECATED: checkCudaErrors( cudaGLRegisterBufferObject(posVbo) );
	(cudaGraphicsGLRegisterBuffer(&res_posvbo, posVbo,
		cudaGraphicsMapFlagsWriteDiscard));

	createVBO(&normalVbo, _maxVerts*sizeof(float)* 3);
	// DEPRECATED: checkCudaErrors(cudaGLRegisterBufferObject(normalVbo));
	(cudaGraphicsGLRegisterBuffer(&res_normvbo, normalVbo,
		cudaGraphicsMapFlagsWriteDiscard));

	{
		// create buffer object
		glGenBuffers(1, &indicesVbo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesVbo);

		// initialize buffer object
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, _maxTriangles * 3 * sizeof(unsigned int), 0, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		glutReportErrors();

		//createVBO(&indicesVbo, _maxTriangles*sizeof(unsigned int)*3);
		(cudaGraphicsGLRegisterBuffer(&res_indicesvbo, indicesVbo,
			cudaGraphicsMapFlagsWriteDiscard));
	}
}

void cspray::renderIsosurface_flat()
{
	glEnable(GL_NORMALIZE);
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);

	glBindBuffer(GL_ARRAY_BUFFER, posVbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalVbo);
	glNormalPointer(GL_FLOAT, sizeof(float)* 3, 0);
	glEnableClientState(GL_NORMAL_ARRAY);

	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_TRIANGLES, 0, totalVerts);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

//半透明绘制
void cspray::renderIsosurface_transparent()
{
	glEnable(GL_NORMALIZE);
	glEnable(GL_LIGHTING);
	glRenderMode(GL_SMOOTH);
	glEnable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);

	glBindBuffer(GL_ARRAY_BUFFER, posVbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalVbo);
	glNormalPointer(GL_FLOAT, sizeof(float)* 3, 0);
	glEnableClientState(GL_NORMAL_ARRAY);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesVbo);

	glColor3f(1.0, 0.0, 0.0);

	//glDrawElements( GL_POINTS, totalIndices, GL_UNSIGNED_INT, 0 );
	glDrawElements(GL_TRIANGLES, totalIndices, GL_UNSIGNED_INT, 0);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void cspray::renderIsosurface_smooth()
{
	glEnable(GL_NORMALIZE);
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glRenderMode(GL_SMOOTH);

	glBindBuffer(GL_ARRAY_BUFFER, posVbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalVbo);
	glNormalPointer(GL_FLOAT, sizeof(float)* 3, 0);
	glEnableClientState(GL_NORMAL_ARRAY);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesVbo);

	glColor3f(1.0, 0.0, 0.0);

	//glDrawElements( GL_POINTS, totalIndices, GL_UNSIGNED_INT, 0 );
	glDrawElements(GL_TRIANGLES, totalIndices, GL_UNSIGNED_INT, 0);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void cspray::rendersphere()
{
	glPushMatrix();
	glEnable(GL_LIGHTING);
	glColor3f(0.0, 0.0, 1.0);
	glTranslatef(solidInitPos.x, solidInitPos.y, solidInitPos.z);
	glutSolidSphere(sphereradius, 100, 100);
	glPopMatrix();
}

void cspray::renderbox_waterfall(int3 minpos, int3 maxpos)
{
	glEnable(GL_LIGHTING);
	glColor3f(0.0, 0.0, 1.0);

	float dx = hparam.cellsize.x;
	float3 boxmin, boxmax;
	//box1
	boxmin = make_float3(0.0f);
	boxmax = make_float3(maxpos.x*dx, minpos.y*dx, maxpos.z*dx);
	glPushMatrix();
	glTranslatef(0.5f*(boxmax.x + boxmin.x), 0.5f*(boxmax.y + boxmin.y), 0.5f*(boxmax.z + boxmin.z));
	glScalef(boxmax.x - boxmin.x, boxmax.y - boxmin.y, boxmax.z - boxmin.z);
	glutWireCube(1.0f);
	glPopMatrix();
	//box2
	boxmin = make_float3(0.0f, minpos.y*dx, 0.0f);
	boxmax = make_float3(maxpos.x*dx, maxpos.y*dx, minpos.z*dx);
	glPushMatrix();
	glTranslatef(0.5f*(boxmax.x + boxmin.x), 0.5f*(boxmax.y + boxmin.y), 0.5f*(boxmax.z + boxmin.z));
	glScalef(boxmax.x - boxmin.x, boxmax.y - boxmin.y, boxmax.z - boxmin.z);
	glutWireCube(1.0f);
	glPopMatrix();
	//box3
	boxmin = make_float3(0.0f, (maxpos.y - 1)*dx, 0.0f);
	boxmax = make_float3(maxpos.x*dx, 1.0f, maxpos.z*dx);
	glPushMatrix();
	glTranslatef(0.5f*(boxmax.x + boxmin.x), 0.5f*(boxmax.y + boxmin.y), 0.5f*(boxmax.z + boxmin.z));
	glScalef(boxmax.x - boxmin.x, boxmax.y - boxmin.y, boxmax.z - boxmin.z);
	glutWireCube(1.0f);
	glPopMatrix();
}

void cspray::rendersphereshader()		/*GY*/
{
	// 	glUseProgram( mProgramSolid );
	// 
	// 	glBindVertexArray( vaoSol );
	// 
	// 	//set uniform variables.
	// 	glm::mat4 model = glm::mat4(1.0f), view, proj;
	// 	setglmLookAtMatrixFromCam( view );
	// 	setglmProjMatrixFromCam( proj );
	// 	mat4 MVP =proj*view*model;		//这里的顺序很重要
	// 	mat4 MV = view*model;
	// 	glUniform1f( glGetUniformLocation(mProgramSolid, "pointScale"), 600.0f / tanf(35.0f*0.5f*(float)M_PI/180.0f) );
	// 	glUniformMatrix4fv(glGetUniformLocation(mProgramSolid, "MVP"), 1, GL_FALSE, &MVP[0][0] );
	// 	glUniformMatrix4fv(glGetUniformLocation(mProgramSolid, "MV"), 1, GL_FALSE, &MV[0][0] );
	// 	glUniformMatrix4fv(glGetUniformLocation(mProgramSolid, "u_Persp"), 1, GL_FALSE, &proj[0][0] );
	// 
	// 	glEnable( GL_DEPTH_TEST );
	// 	glDrawArrays( GL_POINTS, 0, parNumNow );////////////////////////////////////////////
	// 	//////////////////////////////
	// 	glBindVertexArray( 0 );
	// 	glUseProgram(0);
}

void cspray::initSolidBuffers()	///////////////////////////////////////////////////////*G Y*/
{
	// 	glGenVertexArrays(1, &vaoSol);
	// 	glBindVertexArray(vaoSol);
	// 
	// 	glGenBuffers( 1, &vboParPos );
	// 	glBindBuffer( GL_ARRAY_BUFFER, vboSolPos);
	// 	glBufferData( GL_ARRAY_BUFFER, sizeof(float)*3*parNumMax, 0, GL_DYNAMIC_DRAW );///////
	// 	glEnableVertexAttribArray(0);
	// 	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	// 	cudaGLRegisterBufferObject(vboSolPos);
	// 
	// 	glGenBuffers( 1, &vboSolColor );
	// 	glBindBuffer( GL_ARRAY_BUFFER, vboSolColor);
	// 	glBufferData( GL_ARRAY_BUFFER, sizeof(float)*3*parNumMax, 0, GL_DYNAMIC_DRAW );///////////
	// 	glEnableVertexAttribArray(1);
	// 	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	// 	cudaGLRegisterBufferObject(vboSolColor);
	// 
	// 	glBindVertexArray(0);
	// 	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
