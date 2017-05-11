#!/bin/sh
PROJ=-JB-90/40/25/55/4.5i
BOUNDS=-R-130/-60/25/55

gmt pscoast $PROJ $BOUNDS -Dl -N1 -A5000 -M > country.xy
gmt pscoast $RPOJ $BOUNDS -Dl -N2 -A5000 -M > states.xy
gmt pscoast $PROJ $BOUNDS -Dl -N3 -A5000 -M > marine.xy
gmt pscoast $PROJ $BOUNDS -Dl -A5000 -W -M > shorelines.xy
