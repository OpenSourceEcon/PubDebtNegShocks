clear all

global index par siz At calibration column

%  1. Initialize the parameters
% ============================

% 1.1 Initialize model parameters
% -------------------------------

%Size of At, the productivity shock variable
siz.simu = 30000;


% alpha exogenously determined, shared of capital in production function
par.b = 1/3;

%Sigma, mu
par.sigma = 0.2;
par.mu = 3;


% 1.2 Initialize algorithm parameters
% -----------------------------------

siz.herm = 100;             % number of numerical integration nodes
par.order = 3;              %order of polynomial
initials = 'fresh';        
%Type 'fresh' for fixed set of initial values 
               

options = optimset('Display','off','MaxFunEvals',1E5,'MaxIter',...
                  1E5,'TolFun',1E-10,'TolX',1E-10); 

% 1.2 Define all the calibrations in a matrix of coefficients
% -----------------------------------

%Generate grid of growth rates to match for calibration
column = 0;

%New grid, for Rf that goes from -2% to 1%
gridmpk = 1:0.0025:1.04;
gridrf = 0.98:0.005:1.01;
gridmpk = gridmpk.^25;
gridrf = gridrf.^25;



for  EMPK =gridmpk
    for Erf  = gridrf
        if Erf > EMPK + 0.005
            %No risk free rate above risky rate
            break            
        end
        column = column + 1;
        %Matrix that saves all calibrations to solve for
        %Risky rate
        calibration(1, column) = EMPK;
        %Risk free rate
        calibration(2, column) = Erf;
    end
end

%Normal shocks
rng(20110629,'twister');
shock     = randn(siz.simu,1)*par.sigma;


%Finding the endogenous beta, in a model with endowment
for index = 1:column
    
    disp(index);
    
    if  calibration(2,index) > calibration(1,index) + 0.005
        break %No risk-free rate above risky rate            
    end
    
    
    %We choose a starting guess for beta, which we know algebraically should be somewhat close to the following:    
    beta_approx = par.b/(2*calibration(1,index)*(1-par.b)); 
    rangebeta = linspace(beta_approx-0.1,beta_approx+0.1,400);
    
    
    %Start a loop to find beta
    for a=1:size(rangebeta,2)
        
    par.beta(index)=rangebeta(1,a);
        
    %Create "At".    
    At(:,index) = lognrnd(par.mu,par.sigma,[siz.simu,1]);
    At(1,index) = exp(par.mu + ( (par.sigma^2)/2) );
    
            %----Calculate the entire path of K over time:
            
            %Guess initial K
            capital(1,1)=2;
            
            %Endowment is given from its algebraic definition:
            endowment(index) = exp(   log(1-par.b) + (par.b/(1-par.b))*log(par.b) + (1/(1-par.b))*(par.mu+(par.sigma^2)/2) + (par.b/(par.b-1)) * log(calibration(1,index))   );
                        
            %Calculate path of K according to Eq.(10) in our notes.
            for i=1:size(At,1)-1
                wage(i,1) = At(i,index) * (1-par.b) * capital(i,1)^par.b;
                capital(i+1,1)= par.beta(index) * (wage(i,1) + endowment(index) );
                MPK(i,1) = par.b * At(i,index) * capital(i,1)^(par.b-1);
            end
               
    %Find gamma
    par.gamma(index) = (log(calibration(1,index))-log(calibration(2,index)))/(par.sigma^2);

    %We already know the exact value for gamma when R=Rf, so setting that manually.
    if calibration(1,index)==calibration(2,index)
        par.gamma(index)=0;
    end

            %Each guess of beta produces a certain R, which is going to be somewhat different from our calibration. We want the beta with the smallest error.
            ER_approx(a) = mean(MPK(100:size(MPK,1)));
            
            EK_approx(a) = mean(capital(100:size(capital,1)));
            
            EW_approx(a) = mean(wage(100:size(wage,1)));
            
            difR(a) = abs( (ER_approx(a) - calibration(1,index))/calibration(1,index) );                        
            difW(a) = abs( (EW_approx(a) - endowment(index))/endowment(index) );            
                
    end
           
    
%Find the beta with lowest error in difR
value(index)= difR(1);
bestBETA= rangebeta(1);
posBETA=1;

    for i=1:size(rangebeta,2)
        if difR(i)<value(index)
                value(index)=difR(i);
                posBETA=i;
                bestBETA=rangebeta(i);
        end
    end

    
%Save results
par.beta(index)=bestBETA;
par.kss(index)=EK_approx(posBETA); %"par.kss" is the steady state level of capital, E[K]
endowment(index) = EW_approx(posBETA); %Endowment is equal to E[W]
    
%Save R and RF
par.mpk(index)  = calibration(1,index);
par.rfss(index)  =  calibration(2,index);


 %Calculate path of K one more time, with the correct values.
            capital(1,1)=par.kss(index);
            for i=1:size(At,1)-1
                wage(i,1) = At(i,index) * (1-par.b) * capital(i,1)^par.b;
                capital(i+1,1)= par.beta(index) * (wage(i,1) + endowment(index) );
                MPK(i,1) = par.b * At(i,index) * capital(i,1)^(par.b-1);
            end
    

%Interval of K
capital_trunc(:,index)=capital(100:size(capital,1));

end


save storeLargeGridRf par endowment calibration column At gridrf gridmpk siz capital_trunc

%%


clear all
load storeLargeGridRf

global Q K index endowment At2


siz.simu = 5000; % Total amount of random numbers comprising At

options = optimset('Display','off','MaxFunEvals',1E5,'MaxIter',...
                  1E5,'TolFun',1E-10,'TolX',1E-10); 


fun=@solveRF;


%Create very, very large "At".    
rng(20110629,'twister');
shock     = randn(800000,1)*par.sigma;
for index=1:column
At_basedist2 = shock + par.mu;
At2(:,index) = exp(At_basedist2);
At2(1,index) = exp(par.mu + par.sigma^2/2 );
end

    %To solve numerically for the K function, we generate a grid of D and W,
%and then construct a polynomial approximation for Rf.


for index=53 % grid point 53 sets Erf at -1% and Er at 2%
if  calibration(2,index) > calibration(1,index) + 0.005
    %No risk free rate above risky rate
    break            
end        

disp(index);

%Ranges            
rangeQ=[0:0.005:0.8];

rangeK = linspace(min(capital_trunc(:,index)),max(capital_trunc(:,index)),15);

    %Nested Loops over K and Q
    for countK=1:size(rangeK,2)
    K=rangeK(countK);

        for countQ=1:size(rangeQ,2)
        Q=rangeQ(countQ);

        Rfguess=calibration(2,index);
        Rf(countK,countQ,index)= fsolve(fun,Rfguess,options);

        end
    disp("K");
    disp(countK);
    end

    %Generate Polynomial approximation
    clear x y z
    Rfindex=Rf(:,:,index);
   
    count=1;
    for i=1:size(rangeK,2)
        for j=1:size(rangeQ,2)
            x(count)=rangeK(i);
            y(count)=rangeQ(j);
            z(count)=Rfindex(i,j);
            count=count+1;
        end
    end
 
    x=x';
    y=y';
    z=z';
    
    poly(index)=polyfitn([x,y],z,3);
    poly_guide(index)=polyn2sympoly(poly(index));

    %X is Capital, Y is Q (Q is debt/saving)

end

disp("fin");
save RfmatrixPolyFitLargeCubed53 poly poly_guide par siz column calibration gridrf gridmpk endowment

% In case you want to get a written form of the polynomial, run these commands:
% polyn2sympoly(poly(60))
% poly(60).Coefficients


