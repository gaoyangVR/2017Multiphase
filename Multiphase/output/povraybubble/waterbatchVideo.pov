#include "watersetting.inc" 
#include "metals.inc"
#include "stones.inc"
#declare delta = 1;                           
#include concat("waterdata/water", str(frame_number*delta, -5, 0), ".pov")   
#include concat("waterdata/solobubble", str(frame_number*delta, -5, 0), ".pov")                
    
    
     difference{
box {
     < -0.01, -0.01, 0>,  // Near lower left corner
     < 0.39, 0.39,0.8>   // Far upper right corner
     texture {
     T_Grnt7a     // Pre-defined from stones.inc
     scale 20       // Scale by the same amount in all
                    // directions
    }
    rotate z*0     // Equivalent to "rotate <0,20,0>"

    }  
    box {
     < -0.0, -0.0, 0>,  // Near lower left corner
     < 0.38, 0.38,0.9>   // Far upper right corner
     texture {
     T_Grnt7a      // Pre-defined from stones.inc
     scale 20       // Scale by the same amount in all
                    // directions
    }
    rotate x*0     // Equivalent to "rotate <0,20,0>"

    }  
    }