% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=c2notransferonecalib(epsi)

global par indexD S calib;


%Consumption at second period for old generation
xxx(:,1) = (par.b.*exp(epsi + par.mu(calib)).*S(1,:,indexD)).^(1-par.gamma(calib));



% **********************************************************************
% ********************************************************************