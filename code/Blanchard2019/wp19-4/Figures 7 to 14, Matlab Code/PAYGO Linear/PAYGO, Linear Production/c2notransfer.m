% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=c2notransfer(epsi)

global par index S kt;

%M is as defined in Equation(20) of the notes
M = par.b .* (S(1,:,index).^par.epsilon) + (1-par.b);

%Rtplus1 is just R(t+1)

Rtplus1 = (par.b.*exp(epsi+par.mu(index))).*(S(1,:,index).^(par.epsilon-1)) .* M.^((1-par.epsilon)/par.epsilon);
C2 = Rtplus1.*S(1,:,index);

xxx(:,1) = (C2).^(1-par.gamma(index));


% **********************************************************************
% ********************************************************************