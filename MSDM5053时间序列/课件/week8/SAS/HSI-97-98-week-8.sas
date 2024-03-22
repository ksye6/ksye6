option ls=80 ps=80 nodate nonumber;

FILENAME HSI 'C:\ling\teaching\teaching\MSBD5006MSDM5053\Lecture-8\SAS\HSI-97-98.dat';
DATA  HSI; 
INFILE HSI;
INPUT Y $  X;
logx=log( x );
dlogx=dif(logx);
dlogx2=dlogx*dlogx;
obs=_N_;
RUN;

/*
Proc  print data=HSI;
run;

*/

title1 'Hang Seng Index daily (97-98)';



symbol1 c=black v=none  i=join;
symbol2 c=blue v=none i=join;
symbol3 c=red v=none i=join;

proc gplot data=HSI;
plot x*obs=1;
run;

proc gplot data=HSI;
plot dlogx*obs=3;
plot dlogx2*obs=3;
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

/*
Proc print data=b;
run;
*/

 data c;
 set b;
 xx=exp(logx);
 forecast=exp(forecast+STD*STD/2);
 L95=exp(L95);
 U95=exp(u95);
 obs=_N_-1;
 IF _N_ = 1 THEN DELETE;
 run;

/*
proc print data=c;
run;
*/

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



proc autoreg data=HSI;
model dlogx=/ noint   nlag=(2 3 4) archtest

/* GARCH staement */
garch=(q=1, p=1) method=ml; 	

/* EGARCH statement */
 /*garch=(q=1, p=1, type=exp) method=ml; */

/* IGARCH staement */

 /*  garch=(q=1,p=1,type=integ);  */

 /*garch=(q=1,p=1,type=integ,noint); */

/*garch=(q=1,p=1,  mean=linear) method=ml; */

/* garch=(q=1,p=1,  mean=log) method=ml; */

 /*garch=(q=1,p=1,  mean=sqrt) method=ml;  */

/* EGARCH statement */
/*garch=(q=1, p=1, type=exp) method=ml;	*/


output out=r r=yresid  cev=v predicted=p;
run;


 DATA TEMP;
   SET r;   
   pt=p; vt=v;
   IF _N_ = 1 THEN DELETE; 
    eta2=yresid*yresid/v;    
  
  keep vt pt eta2 yresid obs; 
   
run;   

/*
proc print data= r;
run;
*/

proc arima data=Temp;
identify var=yresid nlag=15;
identify var=eta2  nlag=15;
run;




 data rr;
  MERGE temp r; 
  plogx=logx+pt;
  fforecast=exp (plogx+vt/2);
  ll95=exp (plogx-1.96*sqrt(vt));
  uu95=exp (plogx+1.96*sqrt(vt));  
  
 run;
 


DATA BB;
   MERGE c rr;
   KEEP
fforecast
 ll95
 uu95
 forecast
 L95
 U95
xx obs;
run;


title1 'Forecasting by the ARMA and ARMA-GARCH models';
symbol1  c=black v=star i=join;
symbol2  c=red v=none i=join;
symbol3  c=blue v=none i=join; 


proc gplot data=bb;
/*where obs>=360& obs<390; */ 
/*where obs>=200& obs<260; */ 
/*where obs>=260& obs<320; */
/*where obs>=320& obs<360; */
/*where obs>=360& obs<420; */
/*where obs>=420& obs<460; */

where obs>=360& obs<520; 


plot xx*obs=1 
forecast*obs=3     L95*obs=3 U95*obs=3
/
	overlay;
plot xx*obs=1 fforecast*obs=2     ll95*obs=2 uu95*obs=2
/
	overlay;

plot xx*obs=1 fforecast*obs=2     ll95*obs=2 uu95*obs=2
forecast*obs=3     L95*obs=3 U95*obs=3
/
	overlay;

run; 
