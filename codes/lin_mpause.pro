;Function uses the magnetopause model from Lin et al. JGR 2010
;to output return x and y coordinates on the equitorial plane in GSM
; Inputs: 	theta [Rad] - angle between x axis and r where r=sqrt(y^2+z^2)
;           
;           phi [Rad] - angle between the projection of r into the yz plane 
;           and the direction of the positive y axis from 0 to 2!PI in clockwise sense
;           looking from the sun at the earth
;           
;           tilt [Rad] - dipole tilt


FUNCTION lin_mpause, bz,pdyn,pmag,tilt, theta, phi
;result = fltarr(2,356)
;Xm=fltarr(356)
;Ym=fltarr(356)

a0 = 12.544d
a1 = -0.194d
a2 = 0.305d
a3 = 0.0573d
a4 = 2.178d
a5 = 0.0571d
a6 = -0.999d
a7 = 16.473d
a8 = 0.00152d
a9 = 0.381d
a10 = 0.0431d
a11 = -0.00763d
a12 = -0.210d
a13 = 0.0405d
a14 = -4.430d
a15 = -0.636d
a16 = -2.600d
a17 = 0.832d
a18 = -5.328d
a19 = 1.103d
a20 = -0.907d
a21 = 1.450d
sigma = 1.033d

pm = pmag ;magnetic pressure, assumed to be zero for now
pd = pdyn

beta0 = a6+a7*(exp(a8*bz)-1)/(exp(a9*bz)+1)
beta1 = a10
beta2 = a11+a12*tilt
beta3 = a13

dn = a16+a17*tilt+a18*tilt^2
ds = a16-a17*tilt+a18*tilt^2

thetan = a19+a20*tilt
thetas = a19-a20*tilt

en = a21
es = a21

cn = a14*(pdyn)^a15
cs = cn

psi_s = acos(cos(theta)*cos(thetas)+sin(theta)*sin(thetas)*cos(phi-3.*!PI/2.));*180./!DPI
psi_n = acos(cos(theta)*cos(thetan)+sin(theta)*sin(thetan)*cos(phi-!DPI/2.));*180./!DPI

ex = beta0+beta1*cos(phi)+beta2*sin(phi)+beta3*(sin(phi))^2
f = (cos(theta/2.)+a5*sin(2.*theta)*(1.-exp(-1.*theta)))^ex
r0 = a0*(pd+pm)^a1*(1.+a2*(exp(a3*bz)-1.)/(exp(a4*bz)+1.))
r = r0*f+cn*exp(dn*psi_n^en)+cs*exp(ds*psi_s^es)

return, r
END