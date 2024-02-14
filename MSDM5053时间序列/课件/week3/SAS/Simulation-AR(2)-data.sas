/*Demonstrations of Example 3.1 - 3.6*/
Option Pagesize=100 Linesize=120;



Data A;
z=0.0;
z1=0.0;
z2=0.0;
Do n = -100 To 130;		
a=Rannor(8434);	
z2=z1;
z1=z;

z=z1-0.5*z2+ a;	
If n > 0 Then Output;
End;
/*
proc print data=A;
var z1;
run;
*/
proc gplot data=A;
    SYMBOL1 i=join c=red;
	SYMBOL2  c=c;
plot z1*n=1;
run;


proc arima data=A;
identify var=z;
run;

