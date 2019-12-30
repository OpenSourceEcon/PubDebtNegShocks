%-----This code assumes that you have the polyfitn 


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

%----Generate grid of rates to match for calibration
column = 0;

%New grid, for Rf that goes from -2% to 1%
gridmpk = 1:0.0025:1.04;
gridrf = 0.98:0.005:1.01;
gridmpk = gridmpk.^25;
gridrf = gridrf.^25;


%Matrix of calibrations
for  EMPK =gridmpk
    for Erf  = gridrf
        if Erf > EMPK + 0.005
            %No risk free rate above risky rate
            break            
        end
        column = column + 1;

        %Risky rate
        calibration(1, column) = EMPK;
        %Risk free rate
        calibration(2, column) = Erf;
    end
end


%Finding the endogenous mu, in a model with endowment
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
%         At_basedist = shock + par.mu(index);
%         At = exp(At_basedist);
        At(:,index) = lognrnd(par.mu(index),par.sigma,[siz.simulation,1]);
        At(1,index) = exp(par.mu(index) + ( (par.sigma^2)/2) );

        %----Calculate the entire path of K over time:
            
            %Guess initial K
            capital(1,1)=2;
            
            %Determine endowment from algebraic equation
            endowment(index) = (1-par.b) * exp(par.mu(index)+0.5*par.sigma^2);
                        
            %Calculate path of K according to equation of motion
            for i=1:size(At,1)-1
                wage(i,1) = (1-par.b) * At(i,index);
                capital(i+1,1)= par.beta(index) * (wage(i,1) + endowment(index) );
                MPK(i,1) = par.b * At(i,index);
            end
                
    %Solve for gamma from algebraic equation
    par.gamma(index) = (log(calibration(1,index))-log(calibration(2,index)))/(par.sigma^2);

    %We already know the exact value for gamma when R=Rf, so setting that manually.
    if calibration(1,index)==calibration(2,index)
        par.gamma(index)=0;
    end
    
    %Calculate mean of found values, compare empirical R to exogenous R
    R_approx(index) = mean(MPK(100:size(MPK,1)));
    K_approx(index) = mean(capital(100:size(capital,1)));
    difR(index) = (R_approx(index)-calibration(1,index))/calibration(1,index);
                        
%Save results
par.kss(index) = K_approx(index);
par.mpk(index)  = calibration(1,index);
par.rfss(index)  =  calibration(2,index);


            
end

save storeLargeGridRfLinear par kss endowment calibration column  gridrf gridmpk siz At



%%
%Start process for approximation of K
clear all
load storeLargeGridRfLinear
global Q index endowment At2



siz.simu = 5000; % Total amount of random numbers comprising At later
options = optimset('Display','off','MaxFunEvals',1E5,'MaxIter',...
                 1E5,'TolFun',1E-10,'TolX',1E-10); 


%Create very, very large "At".    
rng(20110629,'twister');
shock     = randn(800000,1)*par.sigma;
for index=1:column
At_basedist2 = shock + par.mu(index);
At2(:,index) = exp(At_basedist2);
At2(1,index) = exp(par.mu(index) + par.sigma^2/2 );
end
             
             
%Declare function that solves for Rf
fun=@solveRF;


%To solve numerically for the K function, we generate a grid and then construct a polynomial approximation.

for index=53

    if  calibration(2,index) > calibration(1,index) + 0.005
    %No risk free rate above risky rate
        break            
    end
    
    
disp(index);

%Ranges            
rangeQ=[0:0.005:0.8];

    %Nested Loop over Q

        for countQ=1:size(rangeQ,2)
        Q=rangeQ(countQ);

        Rfguess=calibration(2,index);
        Rf(1,countQ,index)= fsolve(fun,Rfguess,options);
        
        disp("Q");
        disp(countQ);
        end

  
    %Generate Polynomial approximation
    clear x y z
    Rfindex=Rf(:,:,index);
    count=1;

        for j=1:size(rangeQ,2)
            y(count)=rangeQ(j);
            z(count)=Rfindex(1,j);
            count=count+1;
        end

    y=y';
    z=z';
    
    poly(index)=polyfitn(y,z,3);
    poly_guide(index)=polyn2sympoly(poly(index));
    %Y is Q (Q is debt/saving)

end

disp("Fin");

save RfmatrixPolyFitLargeCubedLinear53 poly poly_guide par siz column calibration gridrf gridmpk endowment


% In case you want to get a written form of the polynomial, run these commands:
% polyn2sympoly(poly(60))
% poly(60).Coefficients
        

