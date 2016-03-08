#version 330

layout (location=0) in vec3 pos;
layout (location=1) in vec3 color;

uniform float pointScale;
uniform mat4 MVP;
uniform mat4 MV;

out vec3 fs_PosEye;		//�ӵ�ռ��λ������
out vec3 fscolor;		//�ӵ�ռ��λ������

void main(  void) {
	vec3 posEye = (MV  * vec4(pos.xyz, 1.0f)).xyz;
	float dist = length(posEye);
	fs_PosEye = posEye;

	float pointSize = 3.0f;
	gl_PointSize = pointSize;// * pointScale/ dist;

	gl_Position = MVP*vec4(pos.xyz, 1.0f);
	fscolor = color;
}
