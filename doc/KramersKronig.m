clear;

hbar = 1.0545718e-34; % Planck's reduced constant
c = 299792458; % speed of light;
ec = 1.60217662e-19; % elementary charge


%ImEps = load('optical_elf_Si.txt');
%fermiEnergy = 7.83;

%ImEps = load('optical_elf_Al.txt');
%fermiEnergy = 11.07;

%ImEps = load('optical_elf_Au.txt');
%fermiEnergy = 9.11;

%ImEps = load('optical_elf_SiO2.txt');
%fermiEnergy = 0;

ImEps = load('optical_elf_PMMA.txt');
fermiEnergy = 0;

% Maclaurin's method:
% Ohta, K. and Ishida, H., “Comparison Among Several Numerical Integration Methods for Kramers-Kronig Transformation,” Applied Spectroscopy 42, 952–957 (aug 1988).
%n = 1024;
%W = min(ImEps(1,1))*exp((0:(n+1))/(n-1)*log(ImEps(end,1)/ImEps(1,1)));
W = ImEps(:,1);
n = numel(W);
F = zeros(size(W));
G = exp(interp1(log(ImEps(:,1)),log(ImEps(:,2)),log(W),'linear'));

for i = 1:(n-2)
	wi = W(i);
	if(mod(i,2) ~= 0)
		% odd index
		j = 2:2:n;
	else
		% even index
		j = 1:2:n;
	end
	wj = W(j);
	gj = G(j);
	fj = 0.5*(gj./(wj-wi)+gj./(wj+wi));
	F(i) = 2/pi*(W(i+2)-W(i))*sum(fj)-1;
end
U = -F./(F.^2+G.^2);
V = G./(F.^2+G.^2);

loglog(W,V./(U.^2+V.^2),'-xk');
hold on;
loglog(W,V./((U+1).^2+V.^2),'--xr');
