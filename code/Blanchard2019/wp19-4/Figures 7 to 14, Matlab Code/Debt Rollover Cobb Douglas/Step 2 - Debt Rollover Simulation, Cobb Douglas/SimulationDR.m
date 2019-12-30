%**************************************************************************
%                             
%  Cobb Douglas OLG model with uncertainty and portfolio choice
%--------------------------------------------------------------------------


clear all

load RfmatrixPolyFitLargeCubed53

global  siz par rf I S at calib ti2 tiD dt ti ktsaving ktproduction;


%Parameters constant across all calibrations

siz.herm = 50;             % number of numerical integration nodes
par.order = 5;              %order of polynomial
siz.simuT = 6;           %simulation for at shocks
siz.simu = 1000;
%choice of calibration

calib = 53; % grid point 53 sets Erf at -1% and Er at 2%

%-----Manual parameters------

par.endow(calib)=endowment(calib);
par.beta=par.beta(calib);
par.mu(calib)=par.mu;


%Get calibrations in percentages
calibration_pct=calibration.^(1/25);
calibration_pct=calibration_pct-1;
calibration_pct=calibration_pct*100;

% Pct of initial debt
par.debt = (0:0.01:0.30)';

%Number of different debt cases, 
siz.simuD = size(par.debt,1) ;


%%
% 2.2 Simulate At using that ln(At) is normally distributed
%-------------------------------------------------------

%rng(20110629,'twister')
rng(20110500,'twister')
epsi      = randn(siz.simuT,siz.simu)*par.sigma;

at = exp(epsi + par.mu(calib));
%Set all first-period shocks equal to expectation of At.
at(1,:) = exp(par.mu(calib) + par.sigma^2/2);
at(:,1) = exp(par.mu(calib) + 0.5*par.sigma^2);



%% Case 1: No debt in the economy

 % 3.2 Initialize vectors
% -----------------------
%capital
ktsaving = zeros(siz.simuT,siz.simu, siz.simuD);
ktproduction_base = zeros(siz.simuT,siz.simu, siz.simuD);
ktproduction = zeros(siz.simuT,siz.simu, siz.simuD);
output= zeros(siz.simuT,siz.simu, siz.simuD);

%safe rate
rf      = zeros(siz.simuT+1,siz.simu,siz.simuD);
rftomorrow      = zeros(siz.simuT,siz.simu,siz.simuD);
%consumption second period
c2t      = zeros(siz.simuT,siz.simu,siz.simuD);

%debt
dt      = zeros(siz.simuT,siz.simu,siz.simuD);
dtcompare  = zeros(siz.simuT,siz.simu,siz.simuD);
debt_repay = zeros(siz.simuT,siz.simu,siz.simuD);

%debt over savings
dS      = zeros(siz.simuT,siz.simu,siz.simuD);
%debt over savings, similar variable for the graphs
dStrace      = zeros(siz.simuT,siz.simu,siz.simuD);
%wages
Wnoendow       = zeros(siz.simuT,siz.simu,siz.simuD);
I       = zeros(siz.simuT,siz.simu,siz.simuD);
%savings
S       = zeros(siz.simuT,siz.simu,siz.simuD);

%U
U     = zeros(siz.simuT,siz.simu,siz.simuD);
%Uct
Uc2t     = zeros(siz.simuT,siz.simu,siz.simuD);
%Uc2t
Uct     = zeros(siz.simuT,siz.simu,siz.simuD);

%Utility of old receiving gift 
Uc2tminus1= zeros(1,siz.simu,siz.simuD);

%MPK 
MPK= zeros(siz.simuT,siz.simu,siz.simuD);


% law of motion of Kt+1 = beta(1-alpha)AtKt^alpha when D = 0
%siz.simuD = 1 for zero debt
%------------------------------------------------------

%For period 1, set the production capital left over from T-1 and consequent income today.
    
ktproduction_base(1,:,:)   = par.kss(calib);
ktproduction(1,:,:) = par.kss(calib);
output(1,:,:) = (ktproduction(1,:,:).^par.b) .*at(1,:);

Wnoendow(1,:,:) = ones(1,siz.simu,siz.simuD).*(1-par.b).*at(1,:).*ktproduction_base(1,:,:).^(par.b);
I(1,:,:) = ones(1,siz.simu,siz.simuD).*(1-par.b).*at(1,:).*ktproduction_base(1,:,:).^(par.b) + par.endow(calib);
S(1,:,:) = par.beta*I(1,:,:);

%MPK for period 1
MPK(1,:,:)= par.b.*at(1,:).*ktproduction(1,:,:).^(par.b-1);

for ti  = 1:siz.simuT
    ktsaving(ti,:,1) = par.beta*( (1-par.b).*at(ti,:).*ktproduction_base(ti,:,1).^(par.b) + par.endow(calib));
    ktproduction_base(ti+1,:,1) = ktsaving(ti,:,1);
    output(ti,:,1) = (ktproduction_base(ti,:,1).^par.b) .*at(ti,:);
end

    %Make a Ktproduction vector without the extra row of period 6.
for count=1:size(ktproduction_base,1)-1
    ktproduction(count,:,1)=ktproduction_base(count,:,1);
end 
    
%Generate true solution for Rf when D=0 using the simulation of At
%----------------------------------------------------------------

rf(:,:,1) = par.rfss(calib);

%MPK with zero debt
MPK(:,:,1)= par.b.*at(:,:).*ktproduction(:,:,1).^(par.b-1).*ones(siz.simuT,siz.simu,1);

%Wages with zero debt
Wnoendow(:,:,1) = ones(siz.simuT,siz.simu,1).*(1-par.b).*at(:,:).*ktproduction(:,:,1).^(par.b);
I(:,:,1) = ones(siz.simuT,siz.simu,1).*( (1-par.b).*at(:,:).*ktproduction(:,:,1).^(par.b) + par.endow(calib) );

%Savings with zero debt
S(:,:,1) = par.beta*I(:,:,1);

ktsaving(1,:,1) = par.beta*I(1,:,1);

%generate utility path with zero shadow debt
%=====================

%consumption of the young
Uct(:,:,1) = log((1-par.beta).*I(:,:,1));

%Expected utility of the old
for ti2  = 1:siz.simu   
    for ti = 1:siz.simuT
            Uc2t(ti,ti2,1) = log(numi(@c2rfnodebt,siz,par))./(1-par.gamma(calib));
    end
end

%Realized utility of the old in period 1
Uc2gift(1,:,1) = log(MPK(1,:,1) .* ktproduction(1,:,1));

%Welfare function for representative agent
U = (1-par.beta).*Uct + par.beta.*Uc2t;


%% Case 2: with D =/= 0

%index for gamble failures
gamblefailure = zeros(1,siz.simu,siz.simuD);
gamblefailurepos = zeros(siz.simuT,siz.simu,siz.simuD);

%Set the starting Dt and Kt levels
for tiD  = 1:siz.simuD
     dt(1,:,tiD)   = par.kss(calib)*par.debt(tiD); 
     ktsaving(1,:,tiD) = S(1,:,tiD) - dt(1,:,tiD);
     ktproduction(2,:,tiD) = ktsaving(1,:,tiD);
end

dS(1,:,:)  = dt(1,:,:)./S(1,:,:);
dY(1,:,:) = dt(1,:,:)./output(1,:,:);
dStrace=dS;

for tiD  = 2:siz.simuD
    for ti2  = 1:siz.simu
        
        %Set first-period Rf equal to no-debt Rf
        rf(1,ti2,tiD)=rf(1,ti2,1);
        
        for ti  = 1:siz.simuT-1

            %Calculate next-period dS            
            rf(ti+1,ti2,tiD) = polyvaln(poly(calib),[ktsaving(ti,ti2,tiD),dS(ti,ti2,tiD)]);
            dt(ti+1,ti2,tiD)  = rf(ti+1,ti2,tiD)*dt(ti,ti2,tiD);
            I(ti+1,ti2,tiD) = (1-par.b)*at(ti+1,ti2).*ktsaving(ti,ti2,tiD)^par.b + par.endow(calib);
            S(ti+1,ti2,tiD) = par.beta*I(ti+1,ti2,tiD);                        
            dS(ti+1,ti2,tiD) = dt(ti+1,ti2,tiD)./S(ti+1,ti2,tiD);
    
            %Check whether there is gamble failure, given by future Rf exceeding the threshold
            if rf(ti+1,ti2,tiD) > 1.2824 %Rf=1.2824 means a net rf rate of 1%
                %Alternate threshold: if dS(ti,ti2,tiD) > 0.75  
                
                %We have a failure. Count failure.
                gamblefailure(1,ti2,tiD) = 1;
                gamblefailurepos(ti,ti2,tiD)=1;
                                
                %----First we have to reconstruct the present variables---
                
                %Debt will be repaid in full
                dt(ti,ti2,tiD) = 0;
                
                %Recompute wages, income
                Wnoendowment(ti,ti2,tiD) = (1-par.b)*at(ti,ti2).*ktproduction(ti,ti2,tiD)^par.b;
                I(ti,ti2,tiD) = (1-par.b)*at(ti,ti2).*ktproduction(ti,ti2,tiD)^par.b - (dt(ti-1,ti2,tiD) * rf(ti,ti2,tiD)) + par.endow(calib);                               

                %New path of debt accumulation
                S(ti,ti2,tiD) = par.beta*I(ti,ti2,tiD);
                dS(ti,ti2,tiD) = dt(ti,ti2,tiD)./S(ti,ti2,tiD);
                ktsaving(ti,ti2,tiD)     = par.beta*I(ti,ti2,tiD) - dt(ti,ti2,tiD);
                ktproduction(ti+1,ti2,tiD) = ktsaving(ti,ti2,tiD);
                                  
                %---Now we need to calculate the future---.                               

                rf(ti+1,ti2,tiD) = polyvaln(poly(calib),[ktsaving(ti,ti2,tiD),dS(ti,ti2,tiD)]);
                % law of motion of Dt+1 = Dt*Rft+1
                dt(ti+1,ti2,tiD)  = rf(ti+1,ti2,tiD)*dt(ti,ti2,tiD);
                
                %New Output
                output(ti+1,ti2,tiD) = (ktproduction(ti+1,ti2,tiD)^par.b) *at(ti+1,ti2);

                %wages and income
                Wnoendowment(ti+1,ti2,tiD) = (1-par.b)*at(ti+1,ti2).*ktsaving(ti,ti2,tiD)^par.b;
                I(ti+1,ti2,tiD) = (1-par.b)*at(ti+1,ti2).*ktsaving(ti,ti2,tiD)^par.b + par.endow(calib);

                %Savings
                S(ti+1,ti2,tiD) = par.beta*I(ti+1,ti2,tiD);

                %Saving in form of capital
                ktsaving(ti+1,ti2,tiD) = S(ti+1,ti2,tiD) - dt(ti+1,ti2,tiD);
                ktproduction(ti+2,ti2,tiD) = ktsaving(ti+1,ti2,tiD);

                %debt over saving
                dS(ti+1,ti2,tiD) = dt(ti+1,ti2,tiD)./S(ti+1,ti2,tiD);
                dStrace(ti+1,ti2,tiD)=dS(ti+1,ti2,tiD);
                dY(ti+1,ti2,tiD) = dt(ti+1,ti2,tiD)./output(ti+1,ti2,tiD);

                %MPK with zero debt
                MPK(ti+1,ti2,tiD)= par.b.*at(ti+1,ti2).*ktsaving(ti,ti2,tiD).^(par.b-1);         
                
            else %(No debt failure this period)
                
            %----We continue calculating the future normally----.
                
            % law of motion of Dt+1 = Dt*Rft+1
            dt(ti+1,ti2,tiD)  = rf(ti+1,ti2,tiD)*dt(ti,ti2,tiD);

            %wages and income
            Wnoendowment(ti+1,ti2,tiD) = (1-par.b)*at(ti+1,ti2).*ktsaving(ti,ti2,tiD)^par.b;
            I(ti+1,ti2,tiD) = (1-par.b)*at(ti+1,ti2).*ktsaving(ti,ti2,tiD)^par.b + par.endow(calib);
            
            %Saving
            S(ti+1,ti2,tiD) = par.beta*I(ti+1,ti2,tiD);
            
            %Saving in form of capital
            ktsaving(ti+1,ti2,tiD) = S(ti+1,ti2,tiD) - dt(ti+1,ti2,tiD);
            ktproduction(ti+2,ti2,tiD) = ktsaving(ti+1,ti2,tiD);
            
            %New Output
            output(ti+1,ti2,tiD) = (ktproduction(ti+1,ti2,tiD)^par.b) *at(ti+1,ti2);
                                 
            %debt over saving
            dS(ti+1,ti2,tiD) = dt(ti+1,ti2,tiD)./S(ti+1,ti2,tiD);
            dStrace(ti+1,ti2,tiD)=dS(ti+1,ti2,tiD);
            dY(ti+1,ti2,tiD) = dt(ti+1,ti2,tiD)./output(ti+1,ti2,tiD);
            
            %MPK with zero debt
            MPK(ti+1,ti2,tiD)= par.b.*at(ti+1,ti2).*ktsaving(ti,ti2,tiD).^(par.b-1);   
             
            end 
        
        end
                    
        
        for ti  = siz.simuT

        rf(ti+1,ti2,tiD) = polyvaln(poly(calib),[ktsaving(ti,ti2,tiD),dS(ti,ti2,tiD)]);
        
            if rf(ti+1,ti2,tiD) > 1.2824 %Rf=1.2824 means a net rf rate of 1%
            %Alternate debt failure threshold: if dS(ti,ti2,tiD) > 0.75
                %count failure
                gamblefailure(1,ti2,tiD) = 1;
                gamblefailurepos(ti,ti2,tiD)=1;
                
                %--We have to reconstruct the present variables---
                
                %Debt will be repaid in full
                dt(ti,ti2,tiD) = 0;
                
                %Recompute wages, income
                Wnoendowment(ti,ti2,tiD) = (1-par.b)*at(ti,ti2).*ktproduction(ti,ti2,tiD)^par.b;
                I(ti,ti2,tiD) = (1-par.b)*at(ti,ti2).*ktproduction(ti,ti2,tiD)^par.b - (dt(ti-1,ti2,tiD) * rf(ti,ti2,tiD)) + par.endow(calib);
                
                %New path of debt accumulation
                S(ti,ti2,tiD) = par.beta*I(ti,ti2,tiD) - dt(ti,ti2,tiD);
                dS(ti,ti2,tiD) = dt(ti,ti2,tiD)./S(ti,ti2,tiD);
                ktsaving(ti,ti2,tiD)     = par.beta*I(ti,ti2,tiD) - dt(ti,ti2,tiD);
                ktproduction(ti+1,ti2,tiD) = ktsaving(ti,ti2,tiD);                
                
                
                %---Now we need to calculate the future---.
                          rf(ti+1,ti2,tiD) = polyvaln(poly(calib),[ktsaving(ti,ti2,tiD),dS(ti,ti2,tiD)]);                             
                end
       
    end
        
    end
    
end
    

%Compute Utility for all periods, simulations, and debt levels 
for tiD  = 2:siz.simuD
    for ti2  = 1:siz.simu
        for ti = 1:siz.simuT
            % Expected consumption of the old
            Uc2t(ti,ti2,tiD) = log(numi(@c2rfD,siz,par))./(1-par.gamma(calib));   

            % consumption of the young
            Uct(ti,ti2,tiD) = log(I(ti,ti2,tiD) - S(ti,ti2,tiD));
        end
    end
end
    
%Welfare function for representative agent
U = (1-par.beta).*Uct + par.beta.*Uc2t;

%Utility of old receiving a gift, both in expectation and in realization
for ti2  = 1:siz.simu
    for tiD  = 1:siz.simuD
        ti = 1;
        Uc2tminus1(1,ti2,tiD) = log(numi(@c2rfDgift,siz,par))./(1-par.gamma(calib));
        Uc2gift(1,ti2,tiD) = log( (MPK(ti,ti2,tiD) .* ktproduction(ti,ti2,tiD)) + dt(1,ti2,tiD) );
    end
end


Uc2giftChange = (Uc2gift(1,:,:)-Uc2gift(1,:,1));
meangift = mean(Uc2giftChange,2);


    %---Uctminus1---
    Uctminus1 = log(I(1,1,1)  - S(1,1,1));

    %Welfare function for representative agent
    Uminus1 = (1-par.beta).*Uctminus1 + par.beta.*Uc2tminus1;

    %Uc2tminus1 = (Uc2tminus1 - Uc2tminus1(1,:,1))*100;
    Uminus1 = (Uminus1 - Uminus1(1,:,1) )*100;
    meanUminus1 = mean(Uminus1,2);
    Uctminus1 = 0;


%Turn the zeros in "Ds" into NaNs, for graphing purposes.
for ti2  = 1:siz.simu
    for tiD  = 1:siz.simuD
        for ti = 1:siz.simuT
        
            if dS(ti,ti2,tiD)==0
            dS(ti,ti2,tiD)=NaN;
            end
        
            if dStrace(ti,ti2,tiD)==0
            dStrace(ti,ti2,tiD)=NaN;
            end
            
            if dY(ti,ti2,tiD)==0
            dY(ti,ti2,tiD)=NaN;
            end
            
        end
    end
end

%%
%Separate paths that failed and paths that succeeded
ktS = NaN(siz.simuT,siz.simu,siz.simuD);     
dtS = NaN(siz.simuT,siz.simu,siz.simuD);   
ctS = NaN(siz.simuT,siz.simu,siz.simuD);  
dSS = NaN(siz.simuT,siz.simu,siz.simuD)  ;
dStraceS = NaN(siz.simuT,siz.simu,siz.simuD)  ;
dYS = NaN(siz.simuT,siz.simu,siz.simuD)  ;  
Uc2tS = NaN(siz.simuT,siz.simu,siz.simuD) ; 
MPKS = NaN(siz.simuT,siz.simu,siz.simuD) ;
US = NaN(siz.simuT,siz.simu,siz.simuD)   ;
UctS = NaN(siz.simuT,siz.simu,siz.simuD)   ;

ktF = NaN(siz.simuT,siz.simu,siz.simuD);     
dtF = NaN(siz.simuT,siz.simu,siz.simuD) ;   
ctF = NaN(siz.simuT,siz.simu,siz.simuD) ;  
dSF = NaN(siz.simuT,siz.simu,siz.simuD)  ;  
dStraceF = NaN(siz.simuT,siz.simu,siz.simuD)  ;
dYF = NaN(siz.simuT,siz.simu,siz.simuD)  ;  
MPKF = NaN(siz.simuT,siz.simu,siz.simuD) ;
UF = NaN(siz.simuT,siz.simu,siz.simuD)   ;
Uc2tF = NaN(siz.simuT,siz.simu,siz.simuD)   ;
UctF = NaN(siz.simuT,siz.simu,siz.simuD)   ; 

UFailct = NaN(siz.simuT,siz.simu,siz.simuD)   ;
UFailc2t = NaN(siz.simuT,siz.simu,siz.simuD)   ;
UFail = NaN(siz.simuT,siz.simu,siz.simuD)   ;
USucc = NaN(siz.simuT,siz.simu,siz.simuD)   ;


for tiD  = 1:siz.simuD
    
    for ti2  = 1:siz.simu
        
        Uchange(:,ti2,tiD)= (U(:,ti2,tiD)-U(:,ti2,1))*100;
        
        if gamblefailure(1,ti2,tiD) == 1
            ktF(:,ti2,tiD)      = (ktsaving(:,ti2,tiD) - ktsaving(:,ti2,1))*100;
            rfF(:,ti2,tiD)      = rf(:,ti2,tiD);
            dtF(:,ti2,tiD)      = dt(:,ti2,tiD);
            dSF(:,ti2,tiD)      = dS(:,ti2,tiD);
            dStraceF(:,ti2,tiD)  = dStrace(:,ti2,tiD);
            dYF(:,ti2,tiD)     = dY(:,ti2,tiD);
            MPKF(:,ti2,tiD)     = MPK(:,ti2,tiD).^(1/25);
            UF(:,ti2,tiD)       = (U(:,ti2,tiD) - U(:,ti2,1))*100;
            Uc2tF(:,ti2,tiD)     = (Uc2t(:,ti2,tiD) - Uc2t(:,ti2,1))*100;
            UctF(:,ti2,tiD)      = (Uct(:,ti2,tiD) - Uct(:,ti2,1))*100;
            
            UFail(:,ti2,tiD)      = U(:,ti2,tiD);
            UFailct(:,ti2,tiD)      = Uct(:,ti2,tiD);
            UFailc2t(:,ti2,tiD)      = Uc2t(:,ti2,tiD);
            
           
        elseif  gamblefailure(1,ti2,tiD) == 0
            ktS(:,ti2,tiD)      = (ktsaving(:,ti2,tiD) - ktsaving(:,ti2,1))*100;
            rfS(:,ti2,tiD)      = rf(:,ti2,tiD);
            dtS(:,ti2,tiD)      = dt(:,ti2,tiD);
            dSS(:,ti2,tiD)      = dS(:,ti2,tiD);
            dStraceS(:,ti2,tiD)      = dStrace(:,ti2,tiD);
            dYS(:,ti2,tiD)     = dY(:,ti2,tiD);
            MPKS(:,ti2,tiD)     = MPK(:,ti2,tiD).^(1/25);
            US(:,ti2,tiD)       = (U(:,ti2,tiD) - U(:,ti2,1))*100;
            Uc2tS(:,ti2,tiD)     = (Uc2t(:,ti2,tiD) - Uc2t(:,ti2,1))*100;
            UctS(:,ti2,tiD)      = (Uct(:,ti2,tiD) - Uct(:,ti2,1))*100;

            
            USucc(:,ti2,tiD)      = U(:,ti2,tiD);
            
            
        end
    end
end

%Count deficit gambles that failed
%----------------------------------------------------------------
gambleprob = sum(gamblefailure(1,:,:),2)/siz.simu*100;

%%
% Compute averages and utilities for simulations that failed
%==============================================================

%Check value of safe rate
%------------------------

murfF = nanmean(log(rfF),1);
murfF = nanmean(murfF, 2);
 
% Mean of welfare function and state variable across all simulations
%-------------------------------------------------------------------
meanUF = nanmean(UF,2);
meanUc2tF = nanmean(Uc2tF,2);
meanUctF = nanmean(UctF,2);
meanktF = nanmean(ktF,2);
meanrfF = nanmean(rfF,2);
meanMPKF = nanmean(MPKF,2);
meandtF = nanmean(dtF,2);
meandSF = nanmean(dSF,2);

meanU = nanmean(Uchange,2);

for i=1:size(meanU,1)
    for j=1:size(meanU,3)
        meanUZ(i,j)=meanU(i,1,j);
    end   
end

meanUZminus(:,:)=meanUminus1(:,1,:);


% Percentile of welfare function and state variable across all simulations
%-------------------------------------------------------------------------
CIUF = prctile(UF,[5 95],2);
CIUc2tF = prctile(Uc2tF,[5 95],2);
CIUctF = prctile(UctF,[5 95],2);
CIktF = prctile(ktF,[5 95],2);
CIrfF = prctile(rfF,[5 95],2);
CIMPKF  = prctile(MPKF,[5 95],2);
CIdtF = prctile(dtF,[5 95],2);
CIdSF = prctile(dSF,[5 95],2);

% Percentile of welfare function and state variable across all simulations,
% shade
%-------------------------------------------------------------------------
ARUF = abs(prctile(UF,[5 95],2) - meanUF);
ARUc2tF = abs(prctile(Uc2tF,[5 95],2)- meanUc2tF);
ARUctF = abs(prctile(UctF,[5 95],2)- meanUctF);
ARktF = abs(prctile(ktF,[5 95],2)- meanktF);
ARrfF = abs(prctile(rfF,[5 95],2)- meanrfF);
ARMPKF  = abs(prctile(MPKF,[5 95],2)- meanMPKF);
ARdtF = abs(prctile(dtF,[5 95],2)- meandtF);
ARdSF = abs(prctile(dSF,[5 95],2)- meandSF);

% Compute averages and utilities for simulations that succeeded
%==============================================================

%Check value of safe rate
%------------------------

murfS = nanmean(log(rfS),1);
murfS = nanmean(murfS, 2);
 
% Mean of welfare function and state variable across all simulations
%-------------------------------------------------------------------
meanUS = nanmean(US,2);
meanUc2tS = nanmean(Uc2tS,2);
meanUctS = nanmean(UctS,2);
meanktS = nanmean(ktS,2);
meanrfS = nanmean(rfS,2);
meanMPKS = nanmean(MPKS,2);
meandtS = nanmean(dtS,2);
meandSS = nanmean(dSS,2);

% Percentile of welfare function and state variable across all simulations
%-------------------------------------------------------------------------
CIUS = prctile(US,[5 95],2);
CIUc2tS = prctile(Uc2tS,[5 95],2);
CIUctS = prctile(UctS,[5 95],2);
CIktS = prctile(ktS,[5 95],2);
CIrfS = prctile(rfS,[5 95],2);
CIMPKS  = prctile(MPKS,[5 95],2);
CIdtS = prctile(dtS,[5 95],2);
CIdSS = prctile(dSS,[5 95],2);

% Percentile of welfare function and state variable across all simulations,
% shade
%-------------------------------------------------------------------------
ARUS= abs(prctile(US,[5 95],2) - meanUS);
ARUc2tS = abs(prctile(Uc2tS,[5 95],2)- meanUc2tS);
ARUctS = abs(prctile(UctS,[5 95],2)- meanUctS);
ARktS = abs(prctile(ktS,[5 95],2)- meanktS);
ARrfS = abs(prctile(rfS,[5 95],2)- meanrfS);
ARMPKS  = abs(prctile(MPKS,[5 95],2)- meanMPKS);
ARdtS = abs(prctile(dtS,[5 95],2)- meandtS);
ARdSS = abs(prctile(dSS,[5 95],2)- meandSS);

%% Graphs with shaded area

% Defaults for this blog post
width = 4.5;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 12;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

% we select the printing resolution
iResolution = 500;

year = [0 1 2 3 4];
% set defaut font size

% general graphics, this will apply to any figure you open
% (groot is the default figure object).
set(groot, ...
'DefaultFigureColor', 'w', ...
'DefaultAxesLineWidth', 0.5, ...
'DefaultAxesXColor', 'k', ...
'DefaultAxesYColor', 'k', ...
'DefaultAxesFontUnits', 'points', ...
'DefaultAxesFontSize', 13, ...
'DefaultLineLineWidth', 2, ...
'DefaultTextFontUnits', 'Points', ...
'DefaultTextFontSize', 12, ...
'DefaultAxesBox', 'off', ...
'DefaultLegendBox', 'off', ...
'DefaultAxesTickLength', [0.02 0.02]);
 
% set the tickdirs to go out - need this specific order
set(groot, 'DefaultAxesTickDir', 'out');
set(groot, 'DefaultAxesTickDirMode', 'manual');

% Set the default Size for display
defpos = get(0,'defaultFigurePosition');
set(0,'defaultFigurePosition', [defpos(1) defpos(2) width*100, height*100]);

% Set the defaults for saving/printing to a file
set(0,'defaultFigureInvertHardcopy','on'); % This is the default anyway
set(0,'defaultFigurePaperUnits','inches'); % This is the default anyway
defsize = get(gcf, 'PaperSize');
left = (defsize(1)- width)/2;
bottom = (defsize(2)- height)/2;
defsize = [left, bottom, width, height];
set(0, 'defaultFigurePaperPosition', defsize);
set(0,'DefaultTextInterpreter', 'latex')

%%

%Turn debt share into a 1 to 100 scale, for clearer visibility
dSS=dSS*100;
dSF=dSF*100;
dYS=dYS*100;
dYF=dYF*100;


%Failure vs. no failure
year = (0:1:5)';

%Set the debt level you wish to graph
debtlevel = 16;



%Debt share path: Figure 12
figure
plot(year,dSS(:,:,debtlevel),'b', 'LineWidth',0.1);
hold on
plot(year,dSF(:,:,debtlevel),'r', 'LineWidth',0.1);
xlabel('Time (year)')
ylabel({'Debt share of savings'; '($\%$)'})
xticks(0:1:4)
hAxes=gca;
hAxes.XTickLabel=({'0','25','50','75','100'});
hAxes.TickLabelInterpreter = 'latex';
set(gca, 'FontSize', fsz)
title({['Debt Share of Savings, CD OLG With Uncertainty']; ['ER=',num2str(calibration_pct(1,calib)),'$\%$',' ERf=', num2str(calibration_pct(2,calib)),'$\%$',' $initdebt=$',num2str(par.debt(debtlevel)*100),'$\%$']})
xlim([0 4])

%Aggregate Utility, Many Paths: Figure 14
figure
plot(year,US(:,:,debtlevel),'b', 'LineWidth',0.1);
hold on
plot(year,UF(:,:,debtlevel),'r', 'LineWidth',0.1);
plot (0,meangift(1,1,debtlevel)*100, 'bs')  
xlabel('Time (year)')
ylabel({'Aggregate utility change, relative to no debt'; })
xticks(0:1:4)
hAxes=gca;
hAxes.XTickLabel=({'0','25','50','75','100'});
hAxes.TickLabelInterpreter = 'latex';
set(gca, 'FontSize', fsz)
title({['Change in Utility, CD OLG With Uncertainty']; ['ER=',num2str(calibration_pct(1,calib)),'$\%$',' ERf=',num2str(calibration_pct(2,calib)),'$\%$',' $initdebt=$',num2str(par.debt(debtlevel)*100),'$\%$']})
xlim([0 4])

