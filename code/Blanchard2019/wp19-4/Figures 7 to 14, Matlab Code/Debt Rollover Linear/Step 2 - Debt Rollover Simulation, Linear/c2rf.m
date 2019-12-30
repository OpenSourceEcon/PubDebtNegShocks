% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=c2rf(epsi)

global par calib S ti2;


%Consumption at second period for old generation
xxx(:,1) = (par.b*exp(epsi + par.mu(calib)).*(S(:,ti2,1))).^(1-par.gamma(calib));




% **********************************************************************
% ********************************************************************