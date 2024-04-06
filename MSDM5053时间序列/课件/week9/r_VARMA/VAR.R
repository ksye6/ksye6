#generate II N(0, I_2) rm.

 sig=diag(2) # create the 2-by-2 identity matrix
 xt=rmvnorm(300,rep(0,2),sig) # generate random draws
 #MTSplot(xt) # Obtain time series plots (output not shown)

 par(mfrow=c(2,2))
 x=xt[,1]
 y=xt[,2]
 plot(x,type="l")
 plot(y,type="l")
 
 
 ccm(xt, lag=10)

 mq(xt,10)
 
  sig=diag(3) # Simulation study
  z=rmvnorm(200,rep(0,3),sig)
 
  #MTSplot(z) # Obtain time series plots (output not shown)
  
  
     mq(z,10)

  ###############################################
  phi1=matrix(c(.2,-.6,.3,1.1),2,2) # Input phi_1
  phi1

   sig=matrix(c(1,0.8,0.8,2),2,2) # Input sigma_a
   sig
   
    m1=eigen(phi1) # Obtain eigenvalues & vectors
    m1$values 
    m1$vectors
    
  ################################################################################   
   

      m1 <- VARMAsim(400,arlags=c(1),phi=phi1,sigma=sig) 
      
      x = m1$series[,1]
      y = m1$series[,2]
      
      
      par(mfrow=c(2,2))
      
      plot(x,type="l" )
      plot(y,type="l")
      
      
      z=cbind(x,y) # Create a vector series
      summary(z)

      ccm(z, lag=4)
      
      mq(z,lag=10) # Compute Q(m) statistics
      
      
      m1=VAR(z,2)
      #m1 
      
      ### VAR order specification
      
      names(m2) 
      
      m2=VARorder(z)
      
      
      
      ### Recall VAR estimation
      m1=VAR(z,1)
#      m1 
            #    names(m1)
      resi=m1$residuals
     # resi

      
      mq(resi,adj=4)   # plot not shown
      
      
      ### Chi-square test for parameter constraints
      #m3=VARchi(z,p=1)  #   Number of targeted parameters:  
      #Chi-square test and p-value:  15.16379 0.05603778 
      #The default is      thres =1.645 corrseponding to alpha=0.1
      
      #m3=VARchi(z,p=1,thres=1.96)   #  Number of targeted parameters: 10
      ##Chi-square test and p-value:  31.68739 0.000451394  corrseponding to alpha=0.05
      
      ### Model simplification
      
      
      m1=VAR(z,1) # fit a un-constrained VAR(2) model.
      
      m2=refVAR(m1,thres=1.96) # Model refinement.
      
      
      ### Model checking
      
      MTSdiag(m2,adj=4)
      
      ## Plot not shown 
      
      ##### Prediction
      VARpred(m2,8)  ## Using unconstrained model     orig  125 
      
      
      ## Impulse response functions
      ## Using the simplified model
      Phi=m2$Phi
      Sig=m2$Sigma
     
      
        
      varirf(Phi,Sig)  #  TRUE  Press return to continue  ## Plots not shownSee Figures 2.8 and 2.9 of the book
     
      VARirf(Phi,Sig,orth=F) #    original sigma  Press return to continue  ## Plots not shown
    