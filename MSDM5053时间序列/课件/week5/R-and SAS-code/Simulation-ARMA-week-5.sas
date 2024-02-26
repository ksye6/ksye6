/*Demonstrations of Example 3.1 - 3.6*/
Option Pagesize=100 Linesize=120;





Data B;
a=0;
z=0;
z1=0;
a1=0;
Do n = -100 To 1000;
a=Rannor(8434998);	

z=-0.0*z1+0.8*a1+ a;  	
a1=a;
z1=z;


If n > 0 Then Output;
End;

proc gplot data=B;
    SYMBOL1 i=join;
where n>=1; 
plot z*n=1;
run;

proc arima data=B;
identify var=z minic  perror=(10:10);
estimate q=2 noconstant method=cls;
estimate   p=1 q=1  noconstant method=cls;
forecast out=b lead=4 alpha=0.05 noprint;
run;

data c;
 set b;
 obs=_N_;
 run;
/*
 proc print data=c;
run;
*/
symbol1 i=join c=black v=none;
symbol2 i=join c=blue v=none;
symbol3 i=join c=red v=none;


proc gplot data=c;
/*where obs>=360& obs<390; */ 
where Obs>=900; 

plot z*obs=1 
forecast*obs=3
L95*obs=2
U95*obs=2
/	overlay;
run; 




Data A;
z=0.0;
z1=0;
a1=0;
Do n = -100 To 1000;		
a=Rannor(84988890);	

z=0.5*z1-0.5*a1 + a;	
z1=z;
a1=a;

If n > 0 Then Output;
End;
run;

proc gplot data=A;
    SYMBOL1 i=join c=red;
plot z*n=1;
run;


proc arima data=A;
identify var=z; 

run; 


