#include "colors.inc"          
#include "textures.inc"

camera{
    perspective 
    sky <0,0,1>               
    up <0,0,1>
    right -x*image_width/image_height                  
    //��
    //location <0.8,-0.1,0.58>
    //look_at <0.8,0.5,0.58>
    //Զ
    location <0.8,-1.3,0.55>
   look_at <0.8,0.5,0.55>
    rotate <0,0,0>
    
    //translate <0.1,0.1,0.1>
    }

  
light_source{
    <5,0,5>
    color White
    }                               

sky_sphere{ pigment{ gradient <0,1,0>
                     color_map{ [0   color rgb<1,1,1>         ]//White
                                [0.4 color rgb<0.24,0.34,0.56>*0.8]//~Navy
                                [0.6 color rgb<0.24,0.34,0.56>*0.8]//~Navy
                                [1.0 color rgb<1,1,1>         ]//White
                              }
                     scale 2 }
           } // end of sky_sphere 


global_settings{max_trace_level 5}
  
#declare water_material =  material{
      texture{
        pigment{ rgbf<0.9,0.9,0.9,0.9>}
        finish { ambient 0.9 diffuse 0.5
                 reflection{0.3,0.3 fresnel on}
                 specular 0.6 roughness 0.005
                 conserve_energy
               } // end finish
      } // end of texture
    
      interior{ ior 1.33
                 fade_power 2001
                 fade_distance 0.9
                 fade_color <0.8,0.8,0.8>*0.5
                 caustics 0.16
       } // end of interior
     } // end of material 
     
     
#declare droplettex = texture{ pigment{ rgbf <1,1,1,0.9> } finish {emission <1,1,1>*0.8} }
   
plane{
    z,1.5/60
    pigment{ Gray30 }
    } 
    