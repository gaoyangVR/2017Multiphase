#version 330

layout (location=0) in vec3 pos;
layout (location=1) in vec3 incolor;

uniform mat4 MVP;

out vec3 color;		//视点空间的位置中心

void main(  void) {
	gl_PointSize = 5.0f;
	color = incolor;
// 	if(pos.x>0.5)
// 		color = vec3(0,0,1);
// 	else
// 		color = vec3(0,1,0);
	gl_Position = MVP*vec4(pos.xyz, 1.0f);
}
