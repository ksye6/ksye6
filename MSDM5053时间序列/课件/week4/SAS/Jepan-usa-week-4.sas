      OPTIONS LS=75 PS=300;
      FILENAME AR 'C:\Users\maling\Desktop\MAFS5130\MFS\MSF-data\Jepan-USA.dat';
      
      DATA A;
       INFILE AR;
       INPUT Y $ X  Z W;
       logx=log( x );
	   dlogx=dif(logx);
       date=intnx( 'month', '1jan1970'd, _n_);
       format date monyy.;
      run;

     symbol1 i=join c=black v=none;
     symbol2 i=join c=blue v=none;
     symbol3 i=join c=blue v=none;

	 /*
	 proc print data=a;
	 */
     proc gplot data=A;
           GOPTIONS  
           DEVICE=bmp
           GSfNAME=GRAFOUT1;
      plot x*date=1/ haxis='1jan1970'd to '1jan2000'd by year;
      TITLE1 'Time series plot of Original YEN/USD: 1970-2000 monthly';
     run;

    
    
    proc gplot data=a;
       GOPTIONS 
       DEVICE=bmp
       GSfNAME=GRAFOUT3;
       plot dlogx*date=3 / haxis='1jan1970'd to '1jan2000'd by year;
       TITLE1 'Time series plot of log(YEN/USD)(t)-log(YEN/USD)(t-1): 1970-2000 monthly';
    run;

	proc univariate data=A;             
      histogram dlogx /normal cframe = ligr
                        cfill  = blue kernel(color=black);	  
      var dlogx;	
run;

    proc arima data=A;
       identify var=x  noprint;
       identify var=logx (1)  minic  perror=(10:10) ;
       estimate p=3 noconstant method=cls;      
      /*  estimate q=1 noconstant  method=cls;      *?
	   /* forecast out=res1 lead=0 id=date; */
       forecast outlimit=b lead=12 alpha=0.05 id=date interval=month noprint;


     title1 'Forecasting by the final model';
     data c;
      set b;
       x=exp( logx ); 
       forecast=exp (forecast+STD*STD/2);
       l95=exp (l95);
       u95=exp (u95);
      run;
 
 
proc print data=c;
run;


      symbol1  c=black v=plus;
      symbol2 i=join c=red v=star;
      symbol3 i=join c=blue v=none;; 
     proc gplot data=c;
       GOPTIONS 
       DEVICE=bmp
       GSfNAME=GRAFOUT5;
       where date >= ' 1jan90'd;
       plot x*date=1 forecast*date=2
       l95*date=3 u95*date=3/ 
  	  overlay haxis='1jan90'd to '1jan01'd by year;
    run; 

