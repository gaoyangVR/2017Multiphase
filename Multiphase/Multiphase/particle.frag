#version 330

uniform mat4 u_Persp;

in vec3 fs_PosEye;
in vec3 fscolor;
//u_Persp = gl_ProjectionMatrix

void main(void)
{
	// calculate normal from texture coordinates
	vec3 N;
 	float pointSize = 5.0f;
 
 	N.xy = gl_PointCoord.xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
 
 	float mag = dot(N.xy, N.xy);
 //	if (mag > 1.0) discard;   // kill pixels outside circle
 	N.z = sqrt(1.0-mag);
 
 	//calculate depth
 	vec4 pixelPos = vec4(fs_PosEye + normalize(N)*pointSize,1.0f);
 	vec4 clipSpacePos = u_Persp * pixelPos;
 //	gl_FragDepth = clipSpacePos.z / clipSpacePos.w;

	gl_FragColor = vec4(fscolor,1.0f);
}
