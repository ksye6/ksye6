proc import datafile="C:\ling\teaching\teaching\MSBD5006MSDM5053\Lecture-10\SAS\data1.csv" out=A
DBMS=csv replace;
run;

data A;
set A;
x1=log(z1);
x2=log(z2);
y1=dif(x1);
y2=dif(x2);
z1=z1-1594;
z2=z2-889;
cz=z1-1.396*z2;
t=_n_;
run;

/*
Proc print data=A;
run;*/

proc gplot data=A; 
      symbol1 v = none i = join l = 1; 
      symbol2 v = none i = join l = 2; 
      plot z1 *t = 1  
           z2 *t = 2/overlay; 
   run;

proc varmax data=A;/**Test Cointegration**/
   
   model y1 y2 / p=4 q=1 noint print=(corry diagnose) minic=(p=5 q=5);/**z1 means housing starts number 1994-2008; z2 means housing sold 1994-2008**/
    /*restrict  AR(1,1,1)=0;*/

run;


proc varmax data=A;/**Test Cointegration dftest **/
   model z1 z2 / p=4  noint cointtest=(johansen)/**z1 means housing starts number 1994-2008; z2 means housing sold 1994-2008**/
             print=(estimats);
			 
             cointeg rank=1 normalize=z1;
			 

			 output lead=8;
run;
proc gplot data=A; 
      symbol1 v = none i = join l = 1; 
      plot cz *t = 1  ; 
   run;