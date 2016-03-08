#version 330

in vec3 color;

void main(void)
{
	if(length(color-vec3(0,0,1))<0.001)
		discard;
	gl_FragColor = vec4(color, 1.0f);
}
