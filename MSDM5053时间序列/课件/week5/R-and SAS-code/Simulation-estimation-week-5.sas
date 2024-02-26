/*Demonstrations of Example 3.1 - 3.6*/
Option Pagesize=100 Linesize=120;


Data C;
a=0;
Do n = -100 To 1000;		
a1=a;					
a=Rannor(84348);
z=a - 0.5*a1;
If n > 0 Then Output;
End;


proc gplot data=C;
    SYMBOL1 i=join;
plot z*n=1;
run;

proc arima data=C;
identify var=z nlag=15;
estimate q=1 noconstant method=cls;
estimate p=1 noconstant method=cls;
run;




Data D;
a=0;
z=0;
a1=0;
a2=0;
Do n = -100 To 1000;
a=Rannor(84340);	
z=-0.65*a1-0.24*a2 + a; 
a2=a1;
a1=a;

If n > 0 Then Output;
End;

proc gplot data=D;
    SYMBOL1 i=join;
plot z*n=1;
run;


proc arima data=D;
identify var=z nlag=15   minic  perror=(10:10);
estimate q=2 noconstant method=cls;
estimate   p=3 noconstant method=cls;
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



