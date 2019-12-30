% function output=numerator(epsi);
% Compute the RHS numerator for realization shock
function xxx=denominatorGonRf(epsi)

global par index rf Q;


%numerator fo the right handside equation in Euler equation

% a= par.b.*exp(epsi + par.mu(index));
% b = Q.* (rf - (par.b.*exp(epsi + par.mu(index))));
% c= a + b;
% d = c.^(-par.gamma(index));
% f = a.*e;

xxx = (par.b.*exp(epsi + par.mu(index)) + Q.* (rf - (par.b.*exp(epsi + par.mu(index)) ) )).^(-par.gamma(index));


% **********************************************************************
% ********************************************************************