
option ls=80 ps=80 nodate nonumber;
PROC IMPORT DATAFILE = "C:\Users\maling\Desktop\MAFS5130\MFS\MSF-data\hsi-dji-06-09.csv" OUT = hsi_dji
DBMS=CSV REPLACE;
run;

data hsi_dji;
set hsi_dji(keep=Hsi_Close Dji_Close);
r_hsi=log(Hsi_Close);
r_dji=log(Dji_Close);
x=dif(r_hsi);
y=dif(r_dji);
t=_n_;

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
   
model r_hsi r_dji / p=3 noint      dif(r_hsi(1) r_dji(1))			 

    print=(corry diagnose) minic=(p=5 q=5);

/*restrict  AR(3,2,1)=0, AR(2,1,1)=0, AR(1,1,1)=0, AR(3,2,2)=0;*/

    
	  output lead=8;
	  
   run;

proc varmax data=hsi_dji; 
   

model r_hsi r_dji / q=3 noint      dif(r_hsi(1) r_dji(1))			 

        print=(corry diagnose) minic=(p=5 q=5);
 
    
	  output lead=8;
	  
   run;
