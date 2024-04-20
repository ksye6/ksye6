
option ls=80 ps=80 nodate nonumber;
PROC IMPORT DATAFILE = "C:\ling\teaching\teaching\MSBD5006MSDM5053\Lecture-10\SAS\hsi-dji-06-09.csv" OUT = hsi_dji
DBMS=CSV REPLACE;
run;

data hsi_dji;
set hsi_dji(keep=Hsi_Close Dji_Close);
r_hsi=log(Hsi_Close);
r_dji=log(Dji_Close);
x=dif(r_hsi);
y=dif(r_dji);
t=_n_;
z=r_hsi-1.07*r_dji;
if t>400 then output;
run;
symbol1 i=join c=red v=none;
symbol2 i=join c=blue v=none;
proc gplot data=hsi_dji;

title 'HSI(red) and DJI(blue) from 01/01/2006-03/18/2009'; 
plot Hsi_Close*t=1 Dji_Close*t=2/overlay;

title 'log-return of HSI(red) and DJI(blue) from 01/01/2006-03/18/2009'; 

plot r_hsi*t=1 r_dji*t=2/overlay;
plot x*t=1 y*t=2/overlay;
run;


proc varmax data=hsi_dji; 
   
      model r_hsi r_dji / p=3 noint   

     cointtest=(johansen) 
                 ecm=(rank=1 normalize=r_hsi);           
    
	  output lead=8;
	  
   run;

proc gplot data=hsi_dji;
plot z*t=2/overlay;
run;

/* The following is code for Vector GARCH */

   proc varmax data=hsi_dji; 

   /*   model x y/ noint p=3  garch q=1 form=bekk;  */            
   model x y/  p=1;  garch q=1 p=1  form=bekk; 	 
	  
   run;



  /* model y1 y2 = x1 / p=1 garch=(q=1 p=1  form=bekk); 
   model y1 y2 = x1 / p=1 garch=(q=1 p=1  form=bew); 
   model y1 y2 = x1 / p=1 garch=(q=1 p=1  form=diag); 
   model y1 y2 = x1 / p=1 xlag=1 garch=(q=1);
  Drift In Process means the process has a constant drift before differencing
   */


   /*
     proc print data=hsi_dji;  

      run;
