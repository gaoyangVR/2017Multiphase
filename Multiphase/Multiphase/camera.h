#ifndef CAMERA_H
#define CAMERA_H

#include "utility.h"
#include <math.h>

enum CameraMode{
	CAMERA_NONE = 0,
	CAMERA_MOVE = 1,
	CAMERA_MOVECENTER
};

struct mvec3{
	float x, y, z;
	mvec3(){ x = 0, y = 0, z = 0; }
	mvec3(float _x, float _y, float _z){
		x = _x; y = _y; z = _z;
	}

	mvec3 operator+(mvec3 b){ return mvec3(x + b.x, y + b.y, z + b.z); }
	mvec3 operator-(mvec3 b){ return mvec3(x - b.x, y - b.y, z - b.z); }
	mvec3 operator*(float b){ return mvec3(x*b, y*b, z*b); }
	mvec3 operator*(int b){ return mvec3(x*b, y*b, z*b); }
	mvec3 operator/(float b){ return mvec3(x / b, y / b, z / b); }
	mvec3 operator/(int b){ return mvec3(x / b, y / b, z / b); }
	void operator=(mvec3 b){ x = b.x, y = b.y, z = b.z; }
	void operator+=(mvec3 b){ x += b.x, y += b.y, z += b.z; }
	void operator-=(mvec3 b){ x -= b.x, y -= b.y, z -= b.z; }
	float length(){ return sqrt(x*x + y*y + z*z); }
};

inline mvec3 cross(mvec3 a, mvec3 b)
{
	return mvec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline mvec3 normalize(mvec3 a)
{
	return a / a.length();
}

class Camera
{
public:
	Camera(){};
	~Camera(){};

	void Camera::init(float eyetox, float eyetoy, float eyetoz,
		float camAnglex, float camAngley, float camAnglez,
		float camdis, float fov, int _winw, int _winh, float nearPlane, float farPlane);
	void computeFromPositions();
	void mousemove(int x, int y);
	void mousewheel(float dir);
	void resetCamto(){ cam_to = originalcam_to; computeFromPositions(); };

	mvec3 cam_from, cam_to, cam_angs, cam_up, originalcam_to;
	float cam_dis, cam_fov, cam_aspect, nearplane, farplane;
	int last_x, last_y;
	int winw, winh;
	CameraMode mode;
};

#endif