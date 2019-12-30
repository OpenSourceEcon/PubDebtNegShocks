% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=c2notransferpdv(epsi)

global par index S kt c_nd kt_nodebt z;

%M is as defined in Equation(20) of the notes
M = par.b .* (kt_nodebt(z,c_nd,index).^par.epsilon) + (1-par.b);

%Rtplus1 is just R(t+1)

Rtplus1 = (par.b.*exp(epsi+par.mu(index))).*(kt_nodebt(z,c_nd,index).^(par.epsilon-1)) .* M.^((1-par.epsilon)/par.epsilon);
C2 = Rtplus1.*kt_nodebt(z,c_nd,index);

xxx(:,1) = (C2).^(1-par.gamma(index));


% **********************************************************************
% ********************************************************************