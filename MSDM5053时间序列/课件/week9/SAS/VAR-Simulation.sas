proc iml; 
      sig = {1.0  0.5, 0.5 1.25}; 
      phi = {1.2 -0.5, 0.6 0.3}; 
      /* simulate the vector time series */ 
	  
      call varmasim(y,phi) sigma = sig n = 1000 seed = 34657;    
      cn = {'y1' 'y2'}; 
      create simul1 from y[colname=cn]; 
      append from y; 
   quit;



  
   data simul1; 
      set simul1; 
	  t+1;
 run;
   proc gplot data=simul1; 
      symbol1 v = none i = join l = 1; 
      symbol2 v=none c=blue i = join l = 2; 
      symbol3 v=none c=black i = join l = 2; 
plot y1 * t = 1  
     y2 * t = 2; 
   run;


    proc varmax data=simul1;     
 
      model y1 y2 / p=1 noint print=(corry diagnose) minic=(p=5 q=5); 
      output lead=4; 
   run;



proc iml; 
      sig = {1.0  0.5, 0.5 1.25}; 
      phi = {1.2 -0.5, 0.6 0.3}; 
      theta = {0.5 -0.2, 0.1 0.3}; 
      /* to simulate the vector time series */ 
	  call varmacov(cov, phi, theta, sig) lag=1;
      call varmasim(y,phi,theta) sigma=sig n=1000 seed=34657;    
      cn = {'y1' 'y2'}; 
      create simul3 from y[colname=cn]; 
      append from y;  
      quit;
 run;

 data simul3; 
      set simul3; 
      t+1;       
 run;


proc gplot data=simul3; 
      symbol1 v = none i = join l = 1; 
      symbol2 v=none c=blue i = join l = 2; 
      symbol3 v=none c=black i = join l = 2; 
plot y1 *t = 1  
     y2 *t = 2; 
   run;


proc varmax data=simul3;     
      model y1 y2 / p=1 q=1 noint print=(corry diagnose) minic=(p=5 q=5); 
     output lead=8;
   run;



