
Option Pagesize=100 Linesize=120;


Data A;
r=0;

Do n = -100 To 1000;		
a=Rannor(843409);   

r1=r;
r=0.1+0.9*r1 + a;	

If n > 0 Then Output;
End;


proc gplot data=A;
    SYMBOL1 i=join c=red;
plot r*n=1;
run;




proc arima data=A;
identify var=r ;
estimate p=1  method=cls;
run;



Data B;
a=0;
z=0;
z1=0;
a1=0;
Do n = -100 To 1000;
a1=a;
a=Rannor(8434998);	
z2=z1;
z1=z;

z=z1 +0.0*a1-0.5*z2+ a;  	
 /*z=0.5-0.8*a1 + a;	*/


If n > 0 Then Output;
End;

proc gplot data=B;
    SYMBOL1 i=join;
where n>=1; 
plot z*n=1;
run;



proc arima data=B;
identify var=z nlag=45   minic  perror=(10:10);
estimate p=2  method=cls;
run;



