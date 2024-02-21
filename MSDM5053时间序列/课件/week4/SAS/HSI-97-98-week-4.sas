/*Series Title: Japanese Yen to one U.S. Dollar; Exchange Rate  */
/*data     period AR    last year*/

option ls=80 ps=80 nodate nonumber;

FILENAME HSI 'C:\Users\maling\Desktop\MAFS5130\MFS\MSF-data\HSI-97-98.dat';
DATA  HSI; 
INFILE HSI;
INPUT Y $  X;
logx=log( x );
dlogx=dif(logx);
dlogx2=dlogx*dlogx;
obs=_N_;
RUN;




title1 'Hang Seng Index daily (97-98)';



symbol1 c=black v=none  i=join;
symbol2 c=blue v=none i=join;
symbol3 c=red v=none i=join;

proc gplot data=HSI;
plot x*obs=1;
run;

proc gplot data=HSI;
plot dlogx*obs=3;
run;

proc univariate data=HSI;             
      histogram dlogx /normal cframe = ligr
                        cfill  = blue kernel(color=black);	  
      var dlogx;	
run;


/*an autoregressive model with an order,p,  that minimizes the AIC in the range from 8 to 11.*/

proc arima data=HSI converse;
identify var=logx (1) nlag=15 minic  perror=(8:11);
estimate p=4 noconstant method=cls;
estimate p=(2, 3,4) noconstant method=cls;
/*estimate p=4 noconstant method=uls;*/

/*forecast out=res lead=0;*/
forecast out=b  lead=4 alpha=0.05;
run;

 data c;
 set b;
 xx=exp(logx);
 forecast=exp(forecast+STD*STD/2);
 L95=exp(L95);
 U95=exp(u95);
 obs=_N_-1;
 IF _N_ = 1 THEN DELETE;
 run;

 
proc print data=c;
run;
 
proc gplot data=c;
/*where obs>=360& obs<390; */ 
/*where obs>=200& obs<260; */ 
/*where obs>=260& obs<320; */
/*where obs>=320& obs<360; */
/*where obs>=360& obs<420; */
/*where obs>=420& obs<460; */
where obs>=200; 


plot xx*obs=1 
forecast*obs=3     L95*obs=2 U95*obs=2
/
	overlay;
run; 

