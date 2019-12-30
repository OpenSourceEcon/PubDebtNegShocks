% function output=c2tD(epsi);
% Compute the expected consumption of old generation
function xxx=c2notransferpdvsingle(epsi)

global par index c_nd kt_nodebt;

%Rtplus1 is just R(t+1)

Rtplus1 = par.b.*exp(epsi+par.mu(index)).*(kt_nodebt(c_nd,index).^(par.b-1));
C2 = Rtplus1.*kt_nodebt(c_nd,index);

xxx(:,1) = (C2).^(1-par.gamma(index));


% **********************************************************************
% ********************************************************************