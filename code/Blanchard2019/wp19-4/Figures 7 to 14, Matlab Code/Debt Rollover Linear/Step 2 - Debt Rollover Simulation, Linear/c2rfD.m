% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=c2rfD(epsi)

global par calib dt rf ktsaving ti2 tiD ti;


%Consumption at second period for old generation
xxx(:,1) = ( (par.b*exp(epsi + par.mu(calib))) .*ktsaving(ti,ti2,tiD) + rf(ti+1,ti2,tiD).*dt(ti,ti2,tiD)).^(1-par.gamma(calib));

% **********************************************************************
% ********************************************************************