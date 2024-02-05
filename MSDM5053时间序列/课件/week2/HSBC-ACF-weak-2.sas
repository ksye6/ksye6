option ls=80 ps=80 nodate nonumber;
* Read an Excel spreadsheet using PROC IMPORT;
PROC IMPORT DATAFILE = "C:\ling\teaching\teaching\MSBD5006MSDM5053\Lecture-2\data\HSBC.csv" OUT = HSI 
/*PROC IMPORT DATAFILE = "C:\ling\teaching\teaching\MSBD5006MSDM5053\Lecture-2\data\N225-84-09.csv" OUT = HSI*/	
/*PROC IMPORT DATAFILE = "C:\ling\teaching\teaching\MSBD5006MSDM5053\Lecture-2\data\HSI-06-09.csv" OUT = HSI*/ 	
DBMS=CSV REPLACE;
run;

DATA HSI (RENAME = (Date=Y  Open=x1 High=x2 Low=x3 Close=x4 Volume=x5 AdjClose=x6));
SET HSI;


DATA HSI (RENAME = (x6=x));
SET HSI;

data HSI;
SET HSI;
logx=log(x);
dlogx=100*dif(logx);
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
/*plot dlogx2*obs=3;*/
run;


proc univariate data=HSI;      
       
      histogram dlogx /normal cframe = ligr
                        cfill  = blue kernel(color=black);
	  
      var dlogx;	
run;

/*STATIONARITY=(ADF= AR orders DLAG= s) */

proc arima data=HSI;

/*identify var=dlogx2;*/
identify var=logx (1);
/*stationarity=(adf=(1,2,7,8));*/
/*estimate p=2 noconstant method=uls;
forecast out=b lead=4 alpha=0.05 noprint;*/
run;



run; 
