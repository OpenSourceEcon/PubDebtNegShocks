% function output=numerator(epsi);
% Compute the RHS numerator for realization shock
function xxx=denominatorS(epsi)

global par dt index S rfn;


%numerator fo the right handside equation in Euler equation
xxx =  (par.b*exp(epsi + par.mu(index)).*S  + rfn.*dt).^(-par.gamma(index));







% **********************************************************************
% ********************************************************************