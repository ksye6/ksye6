/*Demonstrations of Example 3.1 - 3.6*/
Option Pagesize=100 Linesize=120;

/*RANDNORMAL( N, Mean, Cov ) ; */

Data A;
z=0.0;
seed=100;
Do n = -100 To 1000;		
/*r1=Rannor(843499998);  */
call rannor(seed, rnormal); 

rt30 = RAND('T',30) ;
rt3 = RAND('T',3) ;

call rancau(seed,rcau);      

If n > 0 Then Output;

end;
run;


proc gplot data=A;
    SYMBOL1 i=join c=red;
plot rnormal*n=1;
plot rt30*n=1;
plot rt3*n=1;
plot rcau*n=1;
run;


proc univariate data=A;             
      histogram rnormal /normal cframe = ligr
                        cfill  = blue kernel(color=black);	  
      var rnormal;	
run;



proc univariate data=A;             
      histogram rt30 /normal cframe = ligr
                        cfill  = blue kernel(color=black);	  
      var rt30;	
run;





proc univariate data=A;             
      histogram rt3 /normal cframe = ligr
                        cfill  = blue kernel(color=black);	  
      var rt3;	
run;



proc univariate data=A;             
      histogram rcau /normal cframe = ligr
                        cfill  = blue kernel(color=black);	  
      var rcau;	
run;













