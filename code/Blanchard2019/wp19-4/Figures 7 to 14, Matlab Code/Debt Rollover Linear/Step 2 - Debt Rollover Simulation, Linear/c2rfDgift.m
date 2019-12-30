% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=c2rfDgift(epsi)

global par calib dt ktsaving ktproduction ti ti2 tiD;

%We use Ktproduction because it is equivalent to Ktsaving of period t-1 = 1-1 = 0
xxx(:,1) = (par.b*exp(epsi + par.mu(calib)) .*ktproduction(1,ti2,tiD) + dt(1,ti2,tiD)).^(1-par.gamma(calib));



% **********************************************************************
% ********************************************************************