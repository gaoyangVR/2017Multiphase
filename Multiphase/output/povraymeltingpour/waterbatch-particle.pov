#include "watersetting.inc" 
#include "metals.inc"

#declare delta = 1;                           
#include concat("waterdata/allparticles", str(frame_number*delta, -5, 0), ".pov")
 