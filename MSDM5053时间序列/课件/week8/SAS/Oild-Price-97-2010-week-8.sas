
option ls=80 ps=80 nodate nonumber;
/*PROC IMPORT DATAFILE = "D:\Users\maling\MSF-data\N225-84-09.csv" OUT = HSI*/
/*PROC IMPORT DATAFILE = "E:\MFS\MSF-data\Dow.csv" OUT = HSI
PROC IMPORT DATAFILE = "D:\Users\maling\MSF-data\zhu_data.csv" OUT = HSI
DBMS=CSV REPLACE;
run;*/

FILENAME indata "C:\ling\teaching\teaching\MSBD5006MSDM5053\Lecture-8\SAS\zhu_data.csv";

DATA HSI;
INFILE indata  EXPANDTABS DSD;	
INPUT date $ x x2;


data HSI;
SET HSI;
logx=log(x);
dlogx=dif(logx);
dlogx2=dlogx*dlogx;
obs=_N_;
RUN;



symbol1 i=join c=black v=none;
symbol2 i=join c=blue v=none;
symbol3 i=join c=red v=none;

proc gplot data=HSI;
plot x*obs=1;
run;

proc gplot data=HSI;
plot dlogx*obs=3;

plot dlogx2*obs=3;
run;

proc univariate data=HSI  normaltest; 

var dlogx;

proc univariate data=HSI noprint;
      histogram dlogx / cframe = ligr
                        cfill  = blue kernel(color=black);
run;


proc arima data=HSI converse;
identify var=logx (1) nlag=15  minic  perror=(10:10);
/*identify var=dlogx2 nlag=15 ;*/
estimate q=8 noconstant method=cls;
estimate q=3 noconstant method=cls;
estimate q=(1 2 3 8) noconstant method=cls;
forecast out=b lead=4 alpha=0.05 noprint;
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

proc gplot data=c;
/*where obs>=360& obs<390; */ 
/*where obs>=200& obs<260; */ 
/*where obs>=260& obs<320; */
/*where obs>=320& obs<360; */
/*where obs>=360& obs<420; */
/*where obs>=420& obs<460; */
where obs>=400; 

plot xx*obs=1 
forecast*obs=3     L95*obs=2 U95*obs=2
/	overlay;
run; 



proc autoreg data=HSI;
model dlogx=/ noint nlag=3  archtest

/* GARCH staement */
  garch=(q=1, p=1) method=ml; 

/* EGARCH statement */
/* garch=(q=1, p=1, type=exp) method=ml;*/

/* IGARCH staement */

/*garch=(q=1,p=1,type=integ);*

 /*garch=(q=1,p=1,type=integ,noint);*/

/*garch=(q=1,p=1,  mean=linear) method=ml; */

 /*garch=(q=1,p=1,  mean=log) method=ml; */

 /*garch=(q=1,p=1,  mean=sqrt) method=ml; */

/* EGARCH statement */
 /*garch=(q=1, p=1, type=exp) method=ml;*/

output out=r r=yresid  cev=v predicted=p;
run;



 DATA TEMP;
   SET r;   
   pt=p; vt=v;
   IF _N_ = 1 THEN DELETE;
   keep vt pt; 
run;   



 data rr;
  MERGE temp r; 
  plogx=logx+pt;
 fforecast=exp (plogx+vt/2);
 ll95=exp (plogx-1.96*sqrt(vt));
 uu95=exp (plogx+1.96*sqrt(vt));
 obs=obs;
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

/*
proc print data=BB; 
*/

title1 'Forecasting by the ARMA and ARMA-GARCH models';
symbol1 i=join c=black v=star;
symbol2 i=join c=red v=none;
symbol3 i=join c=blue v=none; 

proc gplot data=bb;
/*where obs>=360& obs<390; */ 
/*where obs>=200& obs<260; */ 
/*where obs>=260& obs<320; */
/*where obs>=320& obs<360; */
/*where obs>=360& obs<420; */
/*where obs>=420& obs<460; */

where obs>=600; 


plot xx*obs=1 
forecast*obs=3     L95*obs=3 U95*obs=3
/
	overlay;
plot xx*obs=1 fforecast*obs=2   ll95*obs=2 uu95*obs=2
/
	overlay;

plot xx*obs=1 fforecast*obs=2     ll95*obs=2 uu95*obs=2
forecast*obs=3     L95*obs=3 U95*obs=3
/
	overlay;

run; 

