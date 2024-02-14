
Option Pagesize=100 Linesize=120;


Data A;
r=0;
Do n = -100 To 200;		
a=Rannor(843409);   

r1=r;
r=0.1+0.9*r1 + a;	

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



proc univariate data=A;             
      histogram r /normal cframe = ligr
                        cfill  = blue kernel(color=black);	  
      var r;	
run;




proc arima data=A;
identify var=r ;
estimate p=1  method=cls;
forecast out=b lead=4 alpha=0.05 noprint;
run;


 data c;
 set b;
 obs=_N_;
 run;

 
proc print data=c;
run;

 

symbol1  c=black v=none;
symbol2 i=join c=blue v=none;
symbol3 i=join c=red v=none;


proc gplot data=c;
/*where obs>=360& obs<390;  */
where Obs>=100; 

plot r*obs=1 
forecast*obs=3
L95*obs=2
U95*obs=2
/	overlay;
run; 


