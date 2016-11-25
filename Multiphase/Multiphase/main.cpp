#include<Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "spray.h"
#include "GL/glew.h"
#include <GL/freeglut.h>
#include "camera.h"

cspray gspray;
Camera gcamera;

int winw = 600, winh = 600;

void display()
{
	gspray.simstep();
	gspray.render();
}

void keyboard_func(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:
		gspray.averagetime();
		exit(0);
		break;
	case ' ':
		gspray.mpause = !gspray.mpause;
		break;
	case 'R': case 'r':
		gspray.rollrendermode();
		break;
	case 'C': case 'c':
		gspray.rollcolormode(1);
		break;
	case 'V': case 'v':
		gspray.rollcolormode(-1);
		break;
	case 'T': case 't':
		gspray.m_btimer = !gspray.m_btimer;
		break;
	case 'M': case 'm':
		gcamera.resetCamto();
		break;
	case 'p': case'P':
	{
				  if (gspray.renderpartiletype == TYPEAIR)
					  gspray.renderpartiletype = TYPEFLUID;
				  else if (gspray.renderpartiletype == TYPEFLUID)
					  gspray.renderpartiletype = TYPEAIR;
				  break;
	}
	case 'O': case 'o':
		gspray.boutputpovray = !gspray.boutputpovray;
		break;
	case 'j': case 'J':
		gspray.boutputobj = !gspray.boutputobj;
		break;
	case 'I': case 'i':
		gspray.mRecordImage = !gspray.mRecordImage;
		break;
	case 'S': case 's':
		gspray.m_bSmoothMC = !gspray.m_bSmoothMC;
		break;
	case 'D': case 'd':
		gspray.m_DistanceFuncMC = (gspray.m_DistanceFuncMC + 1) % 2;
		break;
	case 'G': case 'g':	//	YLP gas on/off
		gspray.m_bGenGas = !gspray.m_bGenGas;
		break;
	case 'U': case 'u':
		gspray.m_bCPURun = !gspray.m_bCPURun;
		break;
	case 'F': case 'f':
		gspray.m_bFixSolid = !gspray.m_bFixSolid;
		break;
	}
}

void mouse_click(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
		gcamera.mode = CAMERA_MOVE;
	else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
		gcamera.mode = CAMERA_MOVECENTER;
	else
		gcamera.mode = CAMERA_NONE;

	gcamera.last_x = x;
	gcamera.last_y = y;
}

void mouse_move(int x, int y)
{
	gcamera.mousemove(x, y);
}

void mousewheel(int button, int dir, int x, int y)
{
	gcamera.mousewheel(dir*0.1f);
}

void initopengl(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowPosition(620, 0);
	glutInitWindowSize(winw, winh);
	glutCreateWindow("SpraySimulation");
	GLenum err = glewInit();
	if (err != GLEW_OK)
		printf("\nGLEW is not available!!!\n\n");

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard_func);
	glutMouseFunc(mouse_click);
	glutMotionFunc(mouse_move);
	glutMouseWheelFunc(mousewheel);
}

int main(int argc, char **argv)
{
	printf("Begin: "), gspray.PrintMemInfo();

	initopengl(argc, argv);
	printf("initopengl complete.\n");
	gspray.init();
	float cx = gspray.hparam.gmax.x*0.5f;
	float cy = gspray.hparam.gmax.y*0.5f;
	float cz = gspray.hparam.gmax.z*0.5f;
	gcamera.init(cx, cy, cz, 0.0f, 90.0f, 1.0f, 3.0f, 35.0f, winw, winh, 0.1f, 100.0f);

	printf("After initializaion: "), gspray.PrintMemInfo(), printf("\n\n");

	glutMainLoop();

	return 0;
}
