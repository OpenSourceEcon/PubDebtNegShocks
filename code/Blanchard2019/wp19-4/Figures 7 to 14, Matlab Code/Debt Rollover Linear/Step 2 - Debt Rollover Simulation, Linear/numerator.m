% function output=numerator(epsi);
% Compute the RHS numerator for realization shock
function xxx=numerator(epsi)

global par dt index rf S;


%numerator fo the right handside equation in Euler equation
xxx =  par.b.*exp(epsi + par.mu(index)).* (par.b.*exp(epsi + par.mu(index)).*S + rf.*dt).^(-par.gamma(index));







% **********************************************************************
% ********************************************************************