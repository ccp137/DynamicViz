1. Installation

   Unpack the AMMON.NA.CURL.tgz using the following command
   	gunzip -c AMMON.NA.CURL.tgz | tar xf -
   Then 
	cd  AMMON.NA.CURL
   	cd src 
	make all
	cd ..

   This compiles the doquad program and places it into AMMON.NA.CURL/bin

   doquad reads the output of the tomography, e.g., RAYLU/vc_018_001.xyz which had lines
   that give  LAT LON Dispersion LENGTH, e.g.,
19.450 -126.074 2.947 0.00
19.450 -125.121 2.947 0.00
19.450 -124.167 2.947 0.00
   The LAT LON give the coordinate of the tomography cell. Dispersion is
   the dispersion value. LENGTH is the total number of kilometers of rays
   passing through the cell.

   Given a target latitude/longitude pair, doquad searches through the xyz file
   to interpolate a value 

   doquad has several command line options

   doquad -LAT lat -LON lon -F fname -C -3

      The -3 flag says that the data are of the form 
        LAT LON Dispersion and that the test of data for 
	all four bounding cells is not made
	This is used with tomographny results of other researchers

      If the -3 flag is not used, then the program expects  the file given
	by fname, e.g., ../RAYLU/vc_018_001.xyz, to have entries
	of LAT LON Dispersion and LENGTH

	Line 27 of the program doquad (AMMON.NA.CURL/src/doquad) sets
	dl=50. If there are no rays through a cell, the tomography program
        sets a cell dispersion value to the mean of the entire data set at a
	period. This value is a continental average and is not real in any sense 
	for a geographic region.  To understand the interpolation, assume that
        we want the dispersion at point 'x'.  We search through the 'xyz' file
        to find the cell centers that surround the desired 'x'  Denote thes by
	'1', '2',  '3' and '4'  In reality '1' represents a region through which rays pass.
	Now as the tomography inverison is set up, the summation of all ray lengths
	through the cell is performed. The result is the LENGTH for the cell.
	To prevent the lack of data giving a bad result, because of the default 
	behavior, we only perform the interpolation if the LENGTH in each boundiag
	cell is > 'dl' km.  The value of dl=50 is OK for a 100 x 100 km grid.
	If a smaller grid size is used, you may wish to make this smaller. Since the
	inversion program does not provide an error estiamte, this is a crude way
        if saying that the dispersion value is believable. The larger 'dl' is, the
        more constrained the dispersion value.
	
	The -C flag is used to force an interpolation, and is only used for 
	debugging.


             ......1...........2.......
                   '           '
                  '     X       '
                 '               '
             ...3.................4.....
               .                    .


	
	The -C flag is used to force an interpolation, and is only used for 
	debugging.

	
	The -C flag is used to force an interpolation, and is only used for 
	debugging.



2. Software requirements

	It is assumed that Computer PRograms in seismology is installed since you will need the
	programs plotnps, calplt, sprep96, sdisp96, sregn96, slegn96 and sdpeqn96.

	It is also assumed the the ImageMagick package is installed so that you can use the
	program 'convert' to make PNG files

	It is also assumed that the environment parameter GREENDIR points to a Green;s function
        folder, with a subdirectory Models, which has the CUS.mod and WUS.mod files. We use
	these to make predicted dispersion for comparison


3. Getting dispersion values

	cd AMMON.NA.CURL
	cd nGRIDREGION
	FDODISP lat lon    (FDODISP 39 -91 )
 
        or

        LDODIST lat lon    (this gives the dispersion valeus but not the comparison plot)

   FDODISP will give the current SLU dispersion values, and also results from other authors:
   A plot is made showing the following:

	Blue dots - Bensons's CU tomography from ground noise. These are only plotted is the map 
          coordinate is within their tomography range. Note that they have some strange values in 
          the oceans and Gulf of Mexico which are unconstrained in their data set
 
	Yellow dots - Love and Rayleigh phase velocity data from Ekström. 
	  http://www.ldeo.columbia.edu/~ekstrom/Projects/ANT/USANT12.html 
	  Ekström, G., Love and Rayleigh phase-velocity maps, 5-40 s, of the western and central USA 
              from USArray data, Earth Planet. Sci. Lett. (2013), http://dx.doi.org/10.1016/j.epsl.2013.11.022

	Red dots - SLU tomography. This is a combination of group velocities from earthquake data 
	  and group and phase velocity dispersion from ambient noise studies. The ambient noise phase 
	  velocities have been performed for the northestern US, Oklahoma and New Madrid regions. 
	  These are done on a 100 km x 100 km grid
          http://www.eas.slu.edu/eqc/eqc_research/NATOMO/NATOMO2/index.html
           
	Green dots - Global dispersion model GDM52 
	  http://www.ldeo.columbia.edu/~ekstrom/Projects/SWP/GDM52.html. 
	  Ekström, G., A global model of Love and Rayleigh surface wave dispersion 
	      and anisotropy, 25-250 s, Geophy. J. Int., doi:10.1111/j.1365-246X.2011.05225.x 
	  phase and group velocity tomography

	White dots - ASWMS Automated Surface Wave Phase Velocity Measuring System 
	  http://ds.iris.edu/ds/products/aswms/ created by 
	  Ge Jin and James B. Gaherty, Surface wave phase-velocity tomography 
	       based on multichannel cross-correlation, Geophys. J. Int. 2015 201: 1383-1398, 
	       doi: 10.1093/gji/ggv079.

	Red curve - WUS model dispersion curve
	Blue curve - CUS model predicted dispersion

     The purpose of the plot is to shaow all results. On the basis of the plot, you can reject
     dispersion values, e.g., GDM52 Love wave dispersion at periods less than 40 secoodns, and
     SLU NA results for periods less than 3 seconds.  For the SLU tomography,  these odd values
     may be the resulf of some bad data or perhaps the result of the smoothing with period in the
     tomography code.

     The program FDODISP gives the following output:

     Dispersion values

     tomona.disp   - SLU tomography Love/Rayleigh phase/group velocity in SURF96 format
     tomoek.disp   - Ekström tomography from TA data Love/Rayleigh phase velocity
     tomogdm52.disp - Ekström  tomography from gloabl earthquake data
		  	This has very long period dispersion which will constrain the
			upper mantle
     tocolo.disp    - Bensen Love/Rayleigh pahse/group velocity - this is dated and
 			is better in the western US which had TA data at the time. The
			east did not
     tomoasw.disp - ASW Rayleigh wave phase velocity

     The all.disp is a compoait of tomona.dosp tomoek.disp tomogdm52.disp tomoasw.disp

     Graphics
     all.PLT is the plot whosing the Love/Rayleigh phase/group velocities. This is in CPS330
	CALPLOT format.  
     all.eps is the corresponding EPS file
     all.png is the corresponding PNG file for use in the WEB or in Office documents



