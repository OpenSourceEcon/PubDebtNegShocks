% function output=numerator(epsi);
% Compute the RHS numerator for realization shock
function xxx=numeratorS(epsi)

global par dt index rfn S;


%numerator fo the right handside equation in Euler equation
xxx =  par.b.*exp(epsi + par.mu(index)).*(par.b.*exp(epsi + par.mu(index)).*S + rfn.*dt).^(-par.gamma(index));







% **********************************************************************
% ********************************************************************