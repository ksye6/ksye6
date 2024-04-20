
   /*alpha=(-0.4, 0.1), beta=(1,-2)*/
    proc iml; 
      sig = 100*i(2); 
      phi = {-0.2 0.1, 0.5 0.2, 0.8 0.7, -0.4 0.6}; 
      call varmasim(y,phi) sigma = sig n = 1000 initial = 0 
                           seed = 45876;   
      cn = {'y1' 'y2'}; 
      create simul2 from y[colname=cn]; 
      append from y; 
   quit; 
 
   data simul2; 
      set simul2; 
	  z=y1-2*y2;
      t+1;


proc gplot data=simul2; 
      symbol1 v = none i = join l = 1; 
      symbol2 v = none i = join l = 2; 
      plot y1 * t = 1 
           y2 * t = 2/overlay;
      plot   z*t=2; 
   run;


proc varmax data=simul2; 
      model y1 y2 / p=2 noint  cointtest=(johansen=(normalize=y1)) print=(estimats)
        ecm=(rank=1 normalize=y1); 
      output lead=8; 
   run;


