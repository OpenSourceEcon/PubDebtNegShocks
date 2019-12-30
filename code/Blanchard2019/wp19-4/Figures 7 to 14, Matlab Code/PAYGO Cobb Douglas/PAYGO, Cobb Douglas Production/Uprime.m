
% Compute [R(t+1)/C2(t+1)], which is equivalent to U'(c2(t+1))*R(t+1)
function xxx=Uprime(input)

global par index S transfer; 

%M is as defined in Equation(20) of the notes
M = par.b .* (S(2,:,index).^par.epsilon) + (1-par.b);

%Rtplus1 is just R(t+1)
Rtplus1 = (par.b.*exp(input+par.mu(index))).*(S(2,:,index).^(par.epsilon-1)) .* M.^((1-par.epsilon)/par.epsilon);

C2 = (Rtplus1.*S(2,:,index)) + transfer(1,:,index);
xxx(:,1) = 1./C2;
