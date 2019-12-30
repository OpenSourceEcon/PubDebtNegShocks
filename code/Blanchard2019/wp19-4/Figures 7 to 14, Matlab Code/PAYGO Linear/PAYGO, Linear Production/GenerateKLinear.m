
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
    par.beta(index)=0.325; %Beta is exogenous
    
    if  calibration(2,index) > calibration(1,index) + 0.005
        break %No risk-free rate above risky rate            
    end
    
    rng(20110629,'twister')
    shock = randn(siz.simulation,index)*par.sigma; %Shock is the normal distribution with a variance appropriate to our At. We will use it later to compute the distribution of At.
     
    %Mu is defined by the unconditional expectation of R:
    par.mu(index)=log(calibration(1,index))-log(par.b)-0.5*par.sigma^2;    
    

    %---Start a for K
    %Create "At".    
    At_basedist = shock + par.mu(index);
    At = exp(At_basedist);
    At(1,index) = exp(par.mu(index) + ( (par.sigma^2)/2) );
    
            %----Calculate the entire path of K over time:
            
            %Guess initial K
            capital(1,1)=2;
            
            %Determine endowment from algebraic equation
            endowment(index) = (1-par.b) * exp(par.mu(index)+0.5*par.sigma^2);
                        
            %Calculate path of K according to equation of motion
            for i=1:size(At,1)-1
                wage(i,1) = (1-par.b) * At(i,1);
                capital(i+1,1)= par.beta(index) * (wage(i,1) + endowment(index) );
                MPK(i,1) = par.b * At(i,1);
            end
                
%Find gamma
par.gamma(index) = (log(calibration(1,index))-log(calibration(2,index)))/(par.sigma^2);

    %We already know the exact value for gamma when R=Rf, so setting that manually.
    if calibration(1,index)==calibration(2,index)
        par.gamma(index)=0;
    end
    
    %Calculate mean of find values, compare empirical R to exogenous R
    R_approx(index) = mean(MPK(100:size(MPK,1)));
    K_approx(index) = mean(capital(100:size(capital,1)));
    difR(index) = (R_approx(index)-calibration(1,index))/calibration(1,index);
                        
%Save results
par.kss(index) = K_approx(index);
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
%rangeD = par.debt .* (par.kss(index));
rangeD = par.debt .* endowment(index);

rangeW = linspace(min(wage_trunc(:,index)),max(wage_trunc(:,index)),12);

    %Nested Loops over D and W
    for countD=1:size(rangeD,2)
    D=rangeD(countD);

        for countW=1:size(rangeW,2)
        W=rangeW(countW);

        k0=0.1;
        k(countD,countW,index)= fsolve(fun,k0,options);

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
