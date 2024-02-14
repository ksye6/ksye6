/*Demonstrations of Example 3.1 - 3.6*/
Option Pagesize=100 Linesize=120;



Data A;
z=0.0;
r=0;
rz=0;
Do n = -100 To 1000;		
a=Rannor(8434900);   


z1=z;
z=0.0+1.0*z1 + a; 	

r1=r;
r=0.0+0.8*r1 + a;	

rz1=rz;
rz=0.0+1.1*rz1 + a;	

If n > 0 Then Output;
End;

/*
proc print data=A;
var z;
run;
*/


proc gplot data=A;
    SYMBOL1 i=join c=red;
plot r*n=1;
run;


proc gplot data=A;
    SYMBOL1 i=join c=red;
plot z*n=1;
run;

proc gplot data=A;
    SYMBOL1 i=join c=red;
plot rz*n=1;
run;


