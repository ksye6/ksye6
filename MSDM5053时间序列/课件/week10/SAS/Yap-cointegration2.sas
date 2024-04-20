proc import datafile="C:\ling\teaching\teaching\MSBD5006MSDM5053\Lecture-10\SAS\Yapdata2.csv" out=mydata
DBMS=csv replace;
run;

/*
proc print data=mydata;
run;
*/

data mydata;
set mydata;
t=_n_;

cz1=y1-2.839*y2+1.656*y3;


run;

proc gplot data=mydata; 
      symbol1 v = none i = join l = 1; 
      symbol2 v = none i = join l = 2; 
      plot y1*t = 1 
           y2*t = 2
           y3*t=2/overlay;
   run;

   /**y1:Federal Fund rate, y2: 90-day Treasury Bill rate, y3: 1-year Treasury Bill rate**/


proc varmax data=mydata;/**Estimate alpha and beta**/
   model y1 y2 y3 /p=4  cointtest=(johansen=(normalize=y1)) noint
    print=(estimates);        
			  cointeg rank=1;
			   output lead=8;

run;

proc gplot data=mydata; 
      symbol1 v = none i = join l = 1; 
      symbol2 v = none i = join l = 2; 
       plot cz1*t = 1 ;
   run;

/*In the "Cointegration Rank Test Using Trace" table,
   the column Drift In ECM means there is no separate drift 
   in the error correction model(case 1)
   and the column Drift In Process means
   the process has a constant drift before differencing (case 3). 

   The "Cointegration Rank Test Using Trace" table
   shows the trace statistics based on Case 3
   and the "Cointegration Rank Test Using 
   Trace under Restriction" table shows
   the trace statistics based on Case 2.*/
   