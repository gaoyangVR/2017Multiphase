#include <Windows.h>
#include <stdio.h>
#include "camera.h"
#include <GL/freeglut.h>

void Camera::init(float eyetox, float eyetoy, float eyetoz,
	float camAnglex, float camAngley, float camAnglez,
	float camdis, float fov, int _winw, int _winh, float nearPlane, float farPlane)
{
	mode = CAMERA_NONE;
	cam_dis = camdis;
	cam_to.x = eyetox, cam_to.y = eyetoy, cam_to.z = eyetoz;
	originalcam_to.x = eyetox, originalcam_to.y = eyetoy, originalcam_to.z = eyetoz;
	cam_angs.x = camAnglex, cam_angs.y = camAngley, cam_angs.z = camAnglez;
	cam_fov = fov;
	cam_aspect = ((float)_winw) / _winh;
	nearplane = nearPlane;
	farplane = farPlane;
	winw = _winw, winh = _winh;
	cam_up.x = 0.0f, cam_up.y = 0.0f, cam_up.z = 1.0f;
	computeFromPositions();
}

void Camera::computeFromPositions()
{
	cam_from.x = cam_to.x + cam_dis*sin(cam_angs.x * DEGtoRAD) * sin(cam_angs.y * DEGtoRAD) * cam_angs.z;
	cam_from.y = cam_to.y - cam_dis*cos(cam_angs.x * DEGtoRAD) * sin(cam_angs.y * DEGtoRAD) * cam_angs.z;
	cam_from.z = cam_to.z + cam_dis*cos(cam_angs.y * DEGtoRAD) * cam_angs.z;
}


void Camera::mousemove(int x, int y)
{
	int dx = x - last_x;
	int dy = y - last_y;
	if (mode == CAMERA_MOVE)
	{
		cam_angs.x += dx;
		cam_angs.y += dy;

		if (cam_angs.x >= 360.0)	cam_angs.x -= 360.0;
		if (cam_angs.x < 0)		cam_angs.x += 360.0;
		if (cam_angs.y >= 180.0)	cam_angs.y = 180.0;
		if (cam_angs.y <= -180.0)	cam_angs.y = -180.0;
	}
	else if (mode == CAMERA_MOVECENTER)
	{
		//todo: normalize
		mvec3 right = cross(cam_to - cam_from, cam_up);
		cam_to += normalize(right) * dx*(-0.001f);
		cam_to += cam_up * dy*0.001f;
	}

	if (x < 10 || y < 10 || x > winw - 10 || y > winh - 10) {
		glutWarpPointer(winw / 2, winh / 2);
		last_x = winw / 2;
		last_y = winh / 2;
	}
	else {
		last_x = x;
		last_y = y;
	}
	computeFromPositions();
}

void Camera::mousewheel(float dir)
{
	cam_dis += dir;
	if (cam_dis < 0)
		cam_dis = cam_dis - dir;
	computeFromPositions();
}