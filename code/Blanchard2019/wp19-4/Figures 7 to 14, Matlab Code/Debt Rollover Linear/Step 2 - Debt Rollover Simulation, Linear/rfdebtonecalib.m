% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=rfdebtonecalib(epsi)

global par indexD epsilon S calib;


%Consumption at second period for old generation
xxx(:,1) = par.b.*exp(epsi + par.mu(calib)).*(par.b.*exp(epsi + par.mu(calib)).*S(2,:,indexD) + epsilon(1,:,indexD)).^(-par.gamma(calib));




% **********************************************************************
% ********************************************************************