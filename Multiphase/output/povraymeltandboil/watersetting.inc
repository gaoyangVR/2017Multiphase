#include "colors.inc"          
#include "textures.inc"

camera{
    perspective 
    sky <0,0,1>               
    up <0,0,1.1>
    right -x*image_width/image_height                  
     location <0.25,-0.75,0.68>
    look_at <0.25,0.5,0.28>
    rotate <0,0,0>
    
    }
    
    
 light_source{ <5,2,5> color rgb <1,1,1>   
              spotlight
              point_at<0.25,0.25,0>
              radius 30  // hotspot
              tightness 100
              falloff 5
              translate< 0, 0, 0>
            }
     
#declare dx = 1/64;    
difference  {
    box{ <-10,-10,-10>, <10,10,10> }    
    //box{ <dx,-5,dx>, <dx*32-dx,dx*32-dx,0.9> }
    box{ <-5+0.25,-5,dx>, <5+0.25,3,6> }
    pigment { rgb<0.3,0.35,0.76> }                                  //АќЮЇКа
    //texture {F_Glass10}
    //scale <1.35,1.35,1.35>
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
        pigment{ rgbf<0.9,0.9,0.9,0.93>}
        finish { ambient 0.5 diffuse 0.5
                 reflection{0.1,0.1 fresnel on}
                 specular 0.3 roughness 0.005
                 //conserve_energy   
               } // end finish
      } // end of texture
    
      interior{ ior 1.233
                 fade_power 2001
                 fade_distance 0.9
                 fade_color <0.8,0.8,0.8>*0.5
                 caustics 0.06
       } // end of interior
     
     } // end of material  

  
#declare solid_material =  material{
      texture{
        pigment{ rgbf<0.78,0.83,0.9,0.95>}
        finish { ambient 0.9 diffuse 0.5
                 reflection{0.3,0.3 fresnel on}
                 specular 0.6 roughness 0.005
                 conserve_energy
               } // end finish
      } // end of texture
    
     } // end of material 
  
  #declare gas_material =  material{
      texture { 
        finish {ambient 0.25 diffuse 0.18 reflection {0.3,0.3 fresnel on}
                specular 0.6 roughness 0.03
                irid { 0.25 //simulates iridescense = interferences at thin layers (films of oil etc.) 0.25 = part of reflected light
                    thickness 0.1 // of the layer
                    turbulence 0.7 }
        } 
        pigment {color rgbf <1,1,1,0.9>}   
        
      }
       
     } // end of material 
  
#declare droplettex = texture{ pigment{ rgbf <1,1,1,0.9> } finish {emission <1,1,1>*0.8} }
#declare bblTexture = texture {finish {ambient 0.3 diffuse 0.1 reflection 0.55
specular 1 roughness 0.001} pigment {color rgbf <1,1,1,0.9>}}
   
