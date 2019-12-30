
clear all
clc

global kss index par siz At calibration column

%-----------Initializing our model-----------%

%Size of At, the productivity shock variable
siz.simulation = 30000;

%-----Initialize parameters----

%"par.b" is the share of capital, "alpha".
par.b = 1/3;

par.sigma = 0.2;

%Determine the random numbers' seed
rng(20110629,'twister')


%%
%----------PART A, Parameterizing our model: finding Kss, Mu, and Gamma--------------%

%Generate grid of growth rates to match for calibration
column = 0;

%Grid
gridmpk = 1:0.0025:1.04;
gridrf = 0.98:0.0025:1.01;
gridmpk = gridmpk.^25;
gridrf = gridrf.^25;

for MPK = gridmpk
    for RF = gridrf
        if RF > MPK + 0.005
            break %No risk-free rate above risky rate
        end

        column=column+1;
        %Matrix with all calibrations of R and RF:
        calibration(1,column)=MPK;
        calibration(2,column)=RF;
    end
end


%Finding the endogenous beta, in a model with endowment
for index = 1:column
    
    disp(index);
    par.mu(index)=3; %Mu is exogenous
    
    if  calibration(2,index) > calibration(1,index) + 0.005
        break %No risk-free rate above risky rate            
    end
    
    rng(20110629,'twister')
    shock = randn(siz.simulation,index)*par.sigma; %Shock is the normal distribution with a variance appropriate to our At. We will use it later to compute the distribution of At.
     
    %We choose a starting guess for beta, which we know algebraically should be somewhat close to the following:    
    beta_approx = par.b/(2*calibration(1,index)*(1-par.b)); 
    rangebeta = linspace(beta_approx-0.1,beta_approx+0.1,400);
    
    
    %Start a loop to find beta
    for a=1:size(rangebeta,2)
        
    par.beta(index)=rangebeta(1,a);
        
    %Create "At".    
    At_basedist = shock + par.mu(index);
    At = exp(At_basedist);
    At(1,index) = exp(par.mu(index) + ( (par.sigma^2)/2) );
    
            %----Calculate the entire path of K over time:
            
            %Guess initial K
            capital(1,1)=2;
            
            %Endowment is given from its algebraic definition:
            endowment(index) = exp(   log(1-par.b) + (par.b/(1-par.b))*log(par.b) + (1/(1-par.b))*(par.mu(index)+(par.sigma^2)/2) + (par.b/(par.b-1)) * log(calibration(1,index))   );
                        
            %Calculate path of K according to Eq.(10) in our notes.
            for i=1:size(At,1)-1
                wage(i,1) = At(i,1) * (1-par.b) * capital(i,1)^par.b;
                capital(i+1,1)= par.beta(index) * (wage(i,1) + endowment(index) );
                MPK(i,1) = par.b * At(i,1) * capital(i,1)^(par.b-1);
            end
                
%Find gamma
par.gamma(index) = (log(calibration(1,index))-log(calibration(2,index)))/(par.sigma^2);

    %We already know the exact value for gamma when R=Rf, so setting that manually.
    if calibration(1,index)==calibration(2,index)
        par.gamma(index)=0;
    end

            %Each guess of beta produces a certain R, which is going to be somewhat different from our calibration. We want the beta with the smallest error.
            r=capital.^(par.b-1);
            rr(a)=mean(r(100:size(capital,1)));
    
            rf=capital.^(par.b-1);
            rfap1(a)=mean(rf(100:size(capital,1)));
            
            rfap(a)=par.b * rfap1(a) * exp(par.mu(index) + ( (0.5-par.gamma(index)) * par.sigma^2));
            
            R_approx(a) = mean(MPK(100:size(MPK,1)));
            
            K_approx(a) = mean(capital(100:size(capital,1)));
            
            W_approx(a) = mean(wage(100:size(wage,1)));
            
            difR(a) = abs( (R_approx(a) - calibration(1,index))/calibration(1,index) );                        
            difW(a) = abs( (W_approx(a) - endowment(index))/endowment(index) );            
                
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
par.kss(index)=K_approx(posBETA); %"par.kss" is the steady state level of capital, E[K]
endowment(index) = W_approx(posBETA); %Endowment is equal to E[W]
    
%Save R and RF
par.mpk(index)  = calibration(1,index);
par.rfss(index)  =  calibration(2,index);


%Calculate steady state wages with these good values (will need this for polynomial approximation of K in next section)
clear capital
clear wage

capital(1,1)=par.kss(index);
for i=1:size(At,1)-1
    wage(i,1) = At(i,1) * (1-par.b) * capital(i,1)^par.b;
    capital(i+1,1)= par.beta(index) * (wage(i,1) + endowment(index) );
end

wage_trunc(:,index)=wage(500:siz.simulation-1,1);
            

end

save storeLargeGridPoly par kss endowment calibration column At wage_trunc gridrf gridmpk siz

%%
clear all
load storeLargeGridPoly

global W D index endowment


%---------- PART B: SOLVING NUMERICALLY FOR Kt+1, based on At+1----------%

%Initialize the necessary parameters
siz.herm = 100; % Number of numerical integration nodes
par.order = 5; % Order of polynomial
siz.simu = 5000; % Total amount of random numbers comprising At
numvar = 2;

options = optimset('Display','off','MaxFunEvals',1E5,'MaxIter',...
                  1E5,'TolFun',1E-10,'TolX',1E-10); 


fun=@solveK;

%To solve numerically for the K function, we generate a grid of D and W,
%and then construct a polynomial approximation for K.


for index=1:column

if  calibration(2,index) > calibration(1,index) + 0.005
    %No risk free rate above risky rate
    break            
end        

disp(index);

%Ranges            
par.debt = [0,0.1,0.2,0.3,0.4,0.5,0.6];
%rangeD = par.debt .* (endowment(index));
rangeD = par.debt .* (par.kss(index));

rangeW = linspace(min(wage_trunc(:,index)),max(wage_trunc(:,index)),12);

    %Nested Loops over D and W
    for countD=1:size(rangeD,2)
    D=rangeD(countD);

        for countW=1:size(rangeW,2)
        W=rangeW(countW);

        k0=0.1;
%        k(countD,countW,index)= numi(@solveK2,siz,par);
         k(countD,countW,index)= fsolve(fun,k0);

        end

    end


    %Generate Polynomial approximation
    clear x y z
    kindex=k(:,:,index);
   
    count=1;
    for i=1:size(rangeD,2)
        for j=1:size(rangeW,2)
            x(count)=rangeD(i);
            y(count)=rangeW(j);
            z(count)=kindex(i,j);
            count=count+1;
        end
    end
 
    x=x';
    y=y';
    z=z';
    
    poly(index)=polyfitn([x,y],z,3);
    poly_guide(index)=polyn2sympoly(poly(index));

    %X is Debt, Y is Wages

end


save ktmatrixPolyFitLarge poly poly_guide par siz column calibration gridrf gridmpk endowment

% In case you want to get a written form of the polynomial, run these commands:
% polyn2sympoly(poly(60))
% poly(60).Coefficients
