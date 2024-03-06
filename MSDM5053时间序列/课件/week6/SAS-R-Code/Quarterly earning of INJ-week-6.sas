
 title1 'Quarterly earnings of Johnson and Johmson: 1960-1980';
   
   data seriesg;
 INPUT   X;
xlog = log( x );
 obs=_N_;
CARDS;
 .71 
 .63 
 .85 
 .44 
 .61 
 .69 
 .92 
 .55 
 .72 
 .77 
 .92 
 .6 
 .83 
 .8 
 1 
 .77 
 .92 
 1 
 1.24 
 1 
 1.16 
 1.3 
 1.45 
 1.25 
 1.26 
 1.38 
 1.86 
 1.56 
 1.53 
 1.59 
 1.83 
 1.86 
 1.53 
 2.07 
 2.34 
 2.25 
 2.16 
 2.43 
 2.7 
 2.25 
 2.79 
 3.42 
 3.69 
 3.6 
 3.6 
 4.32 
 4.32 
 4.05 
 4.86 
 5.04 
 5.04 
 4.41 
 5.58 
 5.85 
 6.57 
 5.31 
 6.03 
 6.39 
 6.93 
 5.85 
 6.93 
 7.74 
 7.83 
 6.12 
 7.74 
 8.91 
 8.28 
 6.84 
 9.54 
 10.26 
 9.54 
 8.729999 
 11.88 
 12.06 
 12.15 
 8.91 
 14.04 
 12.96 
 14.85 
 9.99 
 16.2 
 14.67 
 16.02 
 11.61 
   ;

   symbol1 i=join c=red v=dot;
   proc gplot data=seriesg;
    plot x * obs = 1;
    plot xlog * obs = 1;
   run;

   
     proc arima data=seriesg;
    
     identify var=xlog(1,4);

      estimate p=(1)(4) noconstant method=cls;
      estimate q=(1)(4) noconstant method=cls;
	
	 
      forecast out=b lead=4 alpha=0.05  noprint;
run; 

/*forcasting by using the final model*/



 data c;
 set b;
 x= exp(xlog); 
 forecast=exp(forecast+STD*STD/2);
 l95=exp(l95);
 u95=exp(u95);
 obs=_N_ ;
 run;

proc print data=c;
run;
 
symbol1 i=join c=black v=star;
symbol2 i=join c=red v=circle;
symbol3 i=join c=blue v=circle; 

proc gplot data=c;
 where obs>=30; 
 plot x*obs=1 forecast*obs=2
    l95*obs=3 u95*obs=3/
	overlay ;

run;  


