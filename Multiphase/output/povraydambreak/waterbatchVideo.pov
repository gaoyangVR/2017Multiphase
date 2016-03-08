#include "watersetting.inc" 
#include "metals.inc"

#declare delta = 1;                           
#include concat("waterdata/water", str(frame_number*delta, -5, 0), ".pov")  
#include concat("waterdata/solobubble", str(frame_number*delta, -5, 0), ".pov")  