% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=c2rfnodebt(epsi)

global par calib ktsaving ti2 ti;

%Consumption at second period for old generation
xxx(:,1) = (par.b*exp(epsi + par.mu(calib)).* (ktsaving(ti,ti2,1).^(par.b-1)) .*(ktsaving(ti,ti2,1))).^(1-par.gamma(calib));




% **********************************************************************
% ********************************************************************