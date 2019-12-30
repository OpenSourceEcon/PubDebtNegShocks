function f = solveK2(epsi)

global W par index endowment D; 

    RHSnum = par.b * exp(epsi+par.mu(index)) * (x^(par.b-1)) * (par.b * exp(epsi+par.mu(index)) * (x^(par.b)) + D)^(-par.gamma(index)); 
    RHSden = (par.b * exp(epsi+par.mu(index)) * x^(par.b) + D)^(1-par.gamma(index));
                    
f = W+endowment(index) - ((1-par.beta(index))/par.beta(index)) * (RHSnum/RHSden)^(-1) - D;                    