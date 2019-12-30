% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=c2debtonecalib(epsi)

global par indexD epsilon S calib;


%Consumption at second period for old generation
xxx(:,1) = (par.b.*exp(epsi + par.mu(calib)).*S(2,:,indexD) + epsilon(1,:,indexD)).^(1-par.gamma(calib));




% **********************************************************************
% ********************************************************************