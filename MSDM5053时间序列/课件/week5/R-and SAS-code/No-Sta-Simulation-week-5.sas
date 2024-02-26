/*Demonstrations of Example 3.1 - 3.6*/
Option Pagesize=100 Linesize=120;


Data A;
z=0.0;
Do n = 0 To 1000;		
a=Rannor(84348899);	
z1=z;
z=0.0+1.0*z1 + a;	
w=z-z1;
If n > 0 Then Output;
End;



/*
proc print data=A;
var z1;
run;
*/

proc gplot data=A;
    SYMBOL1 i=join c=red;
plot z*n=1;
run;


proc arima data=A;
identify var=z nlag=25  stationarity=(adf=(1,2)); 
run;





Data B;
z=0;
z1=0;
Do n =0 To 1000;
a=Rannor(843488);	
z2=z1;
z1=z;
z=1.8*z1- 0.8*z2 + a;
w=z-z1;
If n > 0 Then Output;
End;

proc gplot data=B;
    SYMBOL1 i=join;
plot z*n=1;
run;


proc arima data=B;
identify var=w nlag=15;

estimate p=2 noconstant method=uls;
run;





Data C;
a=0;
z=0;
z1=0;
Do n =0 To 1000;		
a1=a;					
a=Rannor(84348);
z1=z;
z=0.0+z1+a - 0.75*a1;
w=z-z1;
If n > 0 Then Output;
End;


proc gplot data=C;
    SYMBOL1 i=join;
plot z*n=1;
run;

proc arima data=C;
identify var=w nlag=15  minic  perror=(10:10);
estimate q=1 noconstant method=cls;
estimate p=1 noconstant method=cls;
run;





Data D;
a=0;
z=0;
z1=0;
z2=0;
w=0;
Do n =0 To 1000;		
a1=a;					
a=Rannor(843488);
z2=z1;
z1=z;
z=1.9*z1-0.9*z2+a - 0.5*a1;
w=z-z1;
If n > 0 Then Output;
End;




proc gplot data=D;
    SYMBOL1 i=join;
plot z*n=1;
run;

proc arima data=D;
identify var=w nlag=15 minic  perror=(10:10);
run;



