#include "colors.inc"          
#include "textures.inc"
#include "metals.inc"    
#include "glass.inc"
#include "stones.inc" 

camera{
    perspective 
    sky <0,0,1>               
    up <0,0,1>
    right -x*image_width/image_height                  
    location <0.5,-0.35,0.8>
    look_at <0.5,0.35,0.45>
    rotate <0,0,0>
    
    //translate <0.1,0.1,0.1>
    }

  
light_source{
   <3,-3,3>
    color White    

 }    

 light_source{
   <0.5,5,0.5>
    color White    
    shadowless

 }                      
            
    
    
sky_sphere{ pigment{ gradient <0,1,0>
                     color_map{ [0   color rgb<1,1,1>         ]//White
                                [0.4 color rgb<0.24,0.34,0.56>*0.8]//~Navy
                                [0.6 color rgb<0.24,0.34,0.56>*0.8]//~Navy
                                [1.0 color rgb<1,1,1>         ]//White
                              }
                     scale 2 }
           } // end of sky_sphere      
   
  
 /* 
plane{y,10 hollow  // 
      
        texture{ pigment {color rgb<0.1,0.3,0.75>*0.7}
                 #if (version = 3.7 )  finish {emission 1 diffuse 0}
                 #else                 finish { ambient 1 diffuse 0}
                 #end 
               } // end texture 1

        texture{ pigment{ bozo turbulence 0.75
                          octaves 6  omega 0.7 lambda 2 
                          color_map {
                          [0.0  color rgb <0.95, 0.95, 0.95> ]
                          [0.05  color rgb <1, 1, 1>*1.25 ]
                          [0.15 color rgb <0.85, 0.85, 0.85> ]
                          [0.55 color rgbt <1, 1, 1, 1>*1 ]
                          [1.0 color rgbt <1, 1, 1, 1>*1 ]
                          } // end color_map 
                         translate< 3, 0,-1>
                         scale <0.3, 0.4, 0.2>*3
                        } // end pigment
                 #if (version = 3.7 )  finish {emission 1 diffuse 0}
                 #else                 finish { ambient 1 diffuse 0}
                 #end 
               } // end texture 2
       scale 10000
     } //-------------------------------------------------------------
   
       */

global_settings{max_trace_level 5}   


 
      /*  texture{Copper_Metal }   
          
                 finish { ambient 0.3 diffuse 0.5
                 reflection{0.9,0.3 fresnel on}
                 specular 0.60 roughness 0.005
                 conserve_energy
               } // end finish   
      } // end of texture         */
       // end of texture
    
 
#declare water_material =  material{
      texture{
        pigment{ rgbf<0.53,0.53,0.95,1>}
        finish { ambient 0.9 diffuse 0.6
                 reflection{0.3,0.3 fresnel on}
                 specular 0.4 roughness 0.01
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

  
#declare solid_material =  material{
      texture{
        pigment{ rgbf<0.8,0.8,0.8,0.55>}
       //finish {F_Glass3 ambient 0.3}
        
        finish { ambient 0.1 diffuse 0.4
                 reflection{0.3,0.3 fresnel off}
                 specular 0.7 roughness 0.005
                // conserve_energy
               } // end finish
               
      } // end of texture
     //interior {ior 1.2 caustics 1}
     } // end of material 
  
#declare droplettex = texture{ pigment{ rgbf <1,1,1,0.9> } finish {emission <1,1,1>*0.8} }
#declare bblTexture = texture {finish {ambient 0.3 diffuse 0.1 reflection 0.55
specular 1 roughness 0.001} pigment {color rgbf <1,1,1,0.9>}}
   
/*
plane{
    z,1.5*0.5/60   
  //   pigment { White_Marble scale 1.0 
  //  pigment{ Gray30 }  
   
    pigment {checker White Gray}      
 }
 
*/  

cylinder {
        <0.597, 0.38, 0.97>,     // Center of one end
        <0.597, 0.38, 0.87-0.15>,     // Center of other end
        0.0247           // Radius
        open         // Remove end caps    
        texture {New_Brass}
        //finish { ambient 0.3 diffuse 0.6 }
     }        


//在底部绘制一个容器     
#declare dx = 1/64;                           
difference  {
    box{ <0,0,0>, <1,1,0.2> }    
    box{ <dx,dx,dx>, <dx*64-dx,dx*64-dx,0.9> }
    //pigment { rgb<0.4,0.45,0.86> }                          
    pigment { rgb<1,1,1> }  
    //finish {Phong_Shiny}
    texture {T_Stone40}                               
    //finish { ambient 0.25 diffuse 0.7 specular 1 roughness 0.005 reflection 0.1 }
    //texture {F_Glass10}
    //scale <1.35,1.35,1.35>
}    


#declare dropletFinish = finish { ambient 0.15 diffuse 0.5 specular 0.2 roughness 0.005 reflection 0.01 }  