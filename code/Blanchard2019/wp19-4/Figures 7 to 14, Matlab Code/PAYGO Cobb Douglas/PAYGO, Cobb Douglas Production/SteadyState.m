%**************************************************************************
%                   Steady State model: effect of Pay as you go on Cobb Douglas Economy
%                    Epstein Zin utility with large transfers
%
%                   01/12/2018
%**************************************************************************

clear all 
load ktmatrixPolyFitLarge 


global  siz par rf index W transfer S at kt_nodebt kt_debt c_nd c_d;

%Parameters constant across all calibrations

siz.herm = 100;             % number of numerical integration nodes
par.order = 5;              %order of polynomial
siz.simuT = 2000;           %simulation for at shocks
maxiter = 1000;

%% 1. Compute Utility at steady state
par.debt = 20/100; % 5/100 for Figure 9, 20/100 for Figure 10

%First row = steady state, second row = PAYGO

%Technology shocks
at     = zeros(1,siz.simuT,column);
%Endowment
endow     = zeros(1,siz.simuT,column);
%Wages
W      = zeros(2,siz.simuT,column);
%Risk-free rate
rf      = zeros(2,siz.simuT,column);
%Savings
S       = zeros(2,siz.simuT,column);
%U
U     = zeros(2,siz.simuT,column);
%Utility of the old
Uc2t     = zeros(2,siz.simuT,column);
%Uc2t
Uct     = zeros(2,siz.simuT,column);

%Uc2t
MPK    = zeros(2,siz.simuT,column);

%ADD TRANSFER
transfer = zeros(1,siz.simuT,column);

%Extend par.kss so that each shock scenario has one kss (they're all the same, but we need this for the Excel output).
kss = zeros(1,siz.simuT,column);


%Now we start the model for every calibration.

for index = 1:column
    %Fixed endowment, taxes cannot be higher otherwise risk of fiscal crisis
    endow(1,:,index) = endowment(index);
    
    %grids for At
    rng(20110629,'twister')
    epsi = randn(siz.simuT,1)*par.sigma + par.mu(index);
    at(1,:,index) = exp(epsi);
    at(1,1,index) = exp(par.mu(index) + ( (par.sigma^2)/2) ); 

end

clear Uct c1 Uc2t c2today Ec2 U

for index=1:column
kproduction_nodebt(1,index)=par.kss(index);

disp(index);

    for c_nd=1:size(at,2) %Calculate path of K according to Eq.(10).
            wage_nodebt(c_nd,index) = at(1,c_nd,index) * (1-par.b) * (kproduction_nodebt(c_nd,index)^par.b);
            income_nodebt(c_nd,index) = wage_nodebt(c_nd,index)+endowment(index);
            
            kproduction_nodebt(c_nd+1,index)= par.beta(index) * income_nodebt(c_nd,index);
            kt_nodebt(c_nd,index)= kproduction_nodebt(c_nd+1,index);

            mpk_nodebt(c_nd,index) = at(1,c_nd,index) * (par.b) * (kproduction_nodebt(c_nd,index)^(par.b-1));
             
            uct_nodebt(c_nd,index) = (1-par.beta(index)).*log(income_nodebt(c_nd,index)-kt_nodebt(c_nd,index));
            uc2_nodebt(c_nd,index) = par.beta(index)/(1-par.gamma(index)) .*log(numi(@c2notransferpdvsingle,siz,par));

    end    

end


%% Utility at steady state with capital decumulation

%---Define the transfer, D>0---

%transfer = par.debt*endow; %Taxes cannot be higher than endowment, otherwise fiscal crisis.

transfer_amount=par.debt*mean(kt_nodebt,1);
for counter=1:size(calibration,2)
transfer(1,:,counter)=transfer_amount(1,counter);
end


%Technology shocks across time

rng(20110639,'twister')
%shock2 = randn(siz.simuT,1)*par.sigma;

for index=1:column

clear i
clear c_nd


    disp(index);
    %Find what Kss would be in the D>0 world
        
    kproduction_debt(1,index)=par.kss(index);
    
    for c_d=1:size(at,2)
        
        wage_debt(c_d,index) = at(1,c_d,index) * (1-par.b) * (kproduction_debt(c_d,index)^par.b);
        income_debt(c_d,index) = wage_debt(c_d,index)+endowment(index);
       
        kproduction_debt(c_d+1,index) = polyvaln(poly(index),[transfer(1,1,index),wage_debt(c_d,index)]);
        kt_debt(c_d,index)= kproduction_debt(c_d+1,index);
            
        mpk_debt(c_d,index) = at(1,c_d,index).*par.b.*(kproduction_debt(c_d,index)^(par.b-1));
            
        uct_debt(c_d,index) = (1-par.beta(index)).*log(income_debt(c_d,index)-kt_debt(c_d,index)-transfer(1,1,index));
        uc2_debt(c_d,index) = par.beta(index)/(1-par.gamma(index)) .*log(numi(@c2debtpdvsingle,siz,par));
    
    end
        
    
    
end



%Welfare function for representative agent

for i=1:column

    U_nogift(1,i) = log( mpk_nodebt(1,i) * kproduction_nodebt(1,i));
    U_gift(1,i) = log( mpk_debt(1,i) * kproduction_debt(1,i)  + transfer(1,1,i) );
    
    %Now we need to create the gift as the first element of a utility array, and the actual utilities calculated before are period 2 and onwards.
        U_nodebt(1,i)=U_nogift(1,i);
        U_debt(1,i)=U_gift(1,i);

        for j=1:size(at,2)
        U_nodebt(j+1,i) = uct_nodebt(j,i) + uc2_nodebt(j,i);
        U_debt(j+1,i) = uct_debt(j,i) + uc2_debt(j,i);
        end
 
        
        for j=1:size(at,2)
        U_nodebt(j,i) = uct_nodebt(j,i) + uc2_nodebt(j,i);
        U_debt(j,i) = uct_debt(j,i) + uc2_debt(j,i);
        end
        
    for j=1:size(U_debt,1)
    U_change(j,i) =(U_debt(j,i)-U_nodebt(j,i))*100;
    end

end


%Calculate mean utility change of last 500 periods
for i=1:column
   
    temp_range=round(size(at,2)/2);
    temp_uchange(:,1)=U_change(temp_range:size(at,2),i);
    
    meanUchange(i)=mean(temp_uchange);
    
    clear temp_range
    
end


%Calculate PDV of Utility (Unused in paper)

discount=0.8;

for i=1:column    
    sum=0;
    
    for j=1:size(at,2)
        sum=sum + (discount^(j-1))*U_change(j,i);
    end

    PDV(i)=sum;
    PDV(i)=PDV(i)*(1-discount);
        
end

%%

%Separate the PDV of utilities into a matrix of R and Rf (Unused in paper)

countti = 1;
column = 0;
for ti  = gridmpk
    countti2 = 1;    
    for ti2  = gridrf
        if ti2 > ti+0.002
            %No risk free rate above risky rate
            U3DmeanU(countti,countti2) = NaN;
            U3Dpdv(countti,countti2) = NaN;
            countti2 = 1 + countti2;
        else
            column = column + 1;
            U3DmeanU(countti,countti2) = meanUchange(column);
            U3Dpdv(countti,countti2) = PDV(column);
            countti2 = 1 + countti2;
        end
    end
    countti = 1 + countti;

end


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
%Aggregate utility


%Generate variables for graphing Aggregate Utility
col=0;
count=1;
for i=gridrf
    col=col+1;
    row=0;
    for j=gridmpk
        row=row+1;
        arpaZ(count)=U3DmeanU(row,col);
        arpaX(count)=gridrf(col);
        arpaY(count)=gridmpk(row);
        count=count+1;
    end
end



%3D Bars for Aggregate Utility, in percentage points.

arpaX2=arpaX.^(1/25);
arpaY2=arpaY.^(1/25);
arpaX2=(arpaX2-1)*100;
arpaY2=(arpaY2-1)*100;

% Figure 9 or 10 (please set line 22 appropriately).
figure
scatterbar3(arpaX2,arpaY2,arpaZ,0.25)
colorbar
colormap(flipud(parula))
set(gcf,'renderer','zbuffer')

clear ax;
title({['Mean Utility, Cobb Douglas with aggregate risk']; [' $\mu=$',num2str(par.mu(1)), ' $\sigma=$',num2str(par.sigma), ' $D=$',num2str(par.debt*100),'$\%$ of EK']})
xlabel('$Er^f$')
ylabel('Er')    
ax.YAxis.TickLabelFormat = '%.2f';
ax.XAxis.TickLabelFormat = '%.2f';
ax=gca;
ax.YDir = 'reverse';
ytickformat('percentage');
xtickformat('percentage');


gridrf_annualized = gridrf.^(1/25);
gridmpk_annualized = gridmpk.^(1/25);

gridrf_annualized = (gridrf_annualized-1)*100;
gridmpk_annualized = (gridmpk_annualized-1)*100;

gridrf_annualized = round(gridrf_annualized,1);
gridmpk_annualized = round(gridmpk_annualized,1);

