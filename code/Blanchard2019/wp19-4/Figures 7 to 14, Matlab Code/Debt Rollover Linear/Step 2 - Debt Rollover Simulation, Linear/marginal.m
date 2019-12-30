% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=marginal(epsi)

global par index S epsilon;


%Consumption at second period for old generation

xxx = (exp(epsi - 0.5.*par.sigma(index).^2).*par.alpha.*S(2,index).^par.alpha + epsilon(1,index)).^(-1);




% **********************************************************************
% ********************************************************************